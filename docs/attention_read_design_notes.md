# Attention read path — investigation (spec steps §1, §2)

Resolved before any RTL, per the spec's step ordering. Companion to
`kv_cache_design_notes.md`, which owns the write path.

Everything below is sized against the **corrected** token budget: the
read side is DRAM-bandwidth-bound, not compute-bound.

| ctx | cache read/token | token time @1.5 GB/s | cycles @215 MHz | entries | budget |
|---|---|---|---|---|---|
| 4096 | 99 MB | 66 ms | 14.2 M | 917,504 | **15.5 cyc/entry** |
| 40960 | 991 MB | 661 ms | 142.0 M | 9,175,040 | **15.5 cyc/entry** |

An "entry" is one (layer, KV head, position) K line + V line. 256
coordinates in 15.5 cycles means the datapath must sustain **~16
coordinates per cycle**. That single number drives §3.

---

## §1 — Online softmax rescaling

### The rescale is an exact shift, not an `exp()`

The spec is right that `exp(old_max − new_max)` is a real computation
that also multiplies the **128-wide output accumulator**, not just the
scalar sum. But it does not have to be a multiply at all.

Work the softmax in **base 2** instead of base e. Softmax is invariant
under a common change of log base, so fold `log2(e)` into the query
scaling alongside the existing `1/sqrt(d_head)` — both are constants
applied once per query, so this is free.

Then keep the **running max quantized to an integer**:

```
    m       integer,  m >= true running max in base-2 units
    rescale factor = 2^(m_old - m_new),  both integers
                   = an exact arithmetic shift right
```

Consequences:

- **No LUT and no multiplier on the rescale path**, for either the
  scalar sum or the 128-wide accumulator.
- **The rescale contributes exactly zero multiplicative error.** A
  power of two is exact in fixed point. This is the part the spec asked
  to get right, and the right answer is to remove the error source
  structurally rather than budget for it.
- Cost: the running max is up to 1 (base-2) unit above the true max, so
  the largest weight is in `[0.5, 1]` instead of exactly 1. That is one
  bit of dynamic range. Nothing.

### The exp table shrinks to one octave

With `s' − m = −(n + f)`, `n` integer ≥ 0 and `f ∈ [0,1)`:

```
    2^(s'-m) = 2^-n · 2^-f
               ^^^^   ^^^^
               shift  LUT over ONE OCTAVE
```

So the table only ever covers `f ∈ [0,1)` — far easier than SiLU's,
which had to span its whole input range and needed a 257th entry to
avoid a wrap bug. Measured linear-interpolation error:

| entries | step | rel. interp error | effective bits |
|---|---|---|---|
| 128 | 0.0078 | 3.67e-06 | 18.1 |
| **256** | **0.0039** | **9.16e-07** | **20.1** |
| 512 | 0.0020 | 2.29e-07 | 22.1 |

**256 entries + 1 endpoint, 16-bit Q0.16** — 4.1 kbit, against RoPE's
262 kbit. 20 bits of accuracy is far past what V's 3-bit quantization
(~13 dB ≈ 2.2 bits) can use, so the table is not the limiting term.

### Accumulator widths — derived, not assumed

Let `l` = running sum, `o[128]` = running output accumulator, both in
the rotated domain, `L` = context length ≤ 40960.

**Integer bits.** Each term `2^(s'-m) ≤ 1`, so `l ≤ L ≤ 2^15.3` → **16
integer bits**. Hadamard-rotated INT8 gives coordinate magnitude
`≤ ||y|| ≤ N·max|x| = 2^14`, so `o` elements are bounded by
`l · 2^14 ≤ 2^29.3` → **30 integer bits**.

**Fractional bits.** Two error sources:

1. *Accumulation truncation.* Each add truncates ≤ 2^-F; over L adds
   the RMS is `√L · 2^-F/√3`. Worst case is a sharply peaked row where
   `l ≈ 1`. At L=40960 (`√L = 202`), F=24 gives `1.2e-5` relative —
   ~16 clean bits, comfortably below V's quantization floor.
2. *Rescale truncation.* The factor is exact, so the only cost is low
   bits shifted out. Crucially this is **common-mode**: `o` and `l` are
   scaled by the *same* exact factor, and the answer is `o/l`, so the
   factor cancels identically. Only the independent truncation noise in
   each survives, bounded by 2^-F per rescale.

→ **`l`: 16+24 = 40 bits. `o`: 30+18 = 48 bits** (o's 18 fractional
bits give 2^-18 relative at full scale, since its integer range already
carries the magnitude).

**How often does a rescale fire?** For exchangeable scores the number
of running maxima in L draws is the harmonic number `≈ ln L` — about 9
at L=8192. So the 128-wide rescale is *logarithmically rare*. The worst
case is monotonically increasing scores (L rescales), which recency
bias makes not-absurd, so the datapath is sized to absorb it rather
than assume the average.

---

## §2 — GQA scan reuse

**Share the scan. It is not close, and the reason is bandwidth.**

| | one shared scan | two scans (per Q-head) |
|---|---|---|
| DRAM traffic | 991 MB/token @ctx40960 | **1982 MB/token** |
| token latency | 661 ms | **1322 ms** |
| softmax state | 2 × (m, l, o[128]) ≈ **12.3 kbit** | 1 × ≈ 6.1 kbit |
| dot-product compute | 2× | 2× (same) |

Re-scanning doubles the **dominant** resource. Sharing costs ~6 kbit of
extra state and doubles a compute resource that sits **97% idle**. The
compute is identical either way — two Q vectors have to meet every K
either way — so the only real question is whether each K/V line is
*fetched* once or twice, and fetching is the whole cost.

So: one pass over each KV head's cache, holding `NQ_PER_KV = 2`
independent running-softmax states in parallel. Parameterized, since
the ratio is a model property.

Sharing also halves the unpack/reconstruct work, which is real: it runs
once per cache entry instead of once per (entry, Q-head).

---

## §3 — Compute sizing: the inner loop needs **zero multipliers**

The budget says ~16 coordinates/cycle. The interesting result is what
those 16 lanes have to contain.

**The low-bit format turns both dot products into selection networks.**
Position-dependent scales factor out of the inner sum:

*Q·K, with K at 2 bits.* Reconstructed `ŷ_K[i] = (lev_q15[q_i]·vnorm_j)>>15`,
plus the QJL term `±amp_j`. So

```
  score_j = (vnorm_j/2^15)·Σ_i yQ[i]·lev_q15[q_i]  +  amp_j·Σ_i (±yQ[i])
             ^^^^^^^^^^^^^                              ^^^^^^^^^^^^^^^^^
             ONE multiply/position                      pure add/subtract
```

`lev_q15` is **position-independent**, so precompute
`P[i][k] = yQ[i]·lev_q15[k]` — 4 values per coordinate, 512 per Q-head,
once per layer. The inner loop is a **4:1 mux and an adder** per lane.
The QJL contribution is an add/subtract chosen by the stored sign bit —
also multiply-free.

*·V, with V at 3 bits.* With `c_j = (w_j·vnorm_j)>>15`,
`o[i] += c_j·lev_q15[q_ji]` — 8 possible products, so precompute
`c_j·lev_q15[k]` (8 multiplies per position) and the inner loop is an
**8:1 mux and an adder**.

**Multiplier budget per position: ~11** (1 for `vnorm_j`, 1 for
`amp_j`, 8 for the V-side level products, 1 for the weight). Everything
else is muxes and adders. So this warrants a small **purpose-built
selection-and-accumulate datapath**, not a MAC array and not a reused
WS primitive — a 16-lane MAC array would be spending 32 multipliers to
do what 32 muxes do.

Expect **~8–12 DSPs total**, and if synthesis reports DSPs proportional
to lane count, the level-precompute factoring did not happen.

---

## §4 — Consuming `kv_dequant`: interface does **not** match, and why

The spec says to use `kv_dequant` directly on the streaming scan. That
would be **1024× over budget**, and the fix is not to reimplement it.

`kv_dequant` reconstructs `x̂` in the **original** domain, which means a
full 448-cycle inverse butterfly per cache entry:

```
  inverse rotation INSIDE the scan loop   411,041,792 cycles/token @ctx4096
  inverse rotation OUTSIDE (see below)            401,408 cycles/token
  ratio                                              1024x
```

Against a 14.2 M cycle budget, the first is ~29× the *entire* token.

**Both rotations leave the loop, for structural reasons:**

- **K — the rotation cancels.** Scores are inner products and the
  rotation is orthogonal. For Hadamard, `y = H·D·x` gives
  `y_Q · ŷ_K = (HDQ)·(HDK̂) = N·(Q·K̂)` **exactly** (`D²=I`, `HᵀH=N·I`).
  So rotate the *query* once and score directly in the rotated domain.
  K is never inverse-rotated at all.
- **V — the rotation is linear, so it commutes with the sum.**
  `Σ_j w_j·ŷ_Vj = H·D·(Σ_j w_j·V̂_j)`. Accumulate rotated, inverse-rotate
  **once** at the end of the scan instead of once per position.

Total read-side rotation work becomes 896 rotations/token (one per
layer per Q-head, each way) — **1.4% of budget**, versus 2900% for the
per-entry form.

**Therefore:** factor `kv_dequant` into `kv_unpack` (words → unpacked
indices → level reconstruction → rotated-domain coordinate stream) plus
the existing inverse butterfly. The scan consumes `kv_unpack`;
`kv_dequant` keeps its current behaviour by instantiating it, so
`tb_kv_dequant` stays green and nothing verified is disturbed. That
honours "don't reimplement" — it is a refactor, not a second copy.

Even better, per §3 the scan does not want reconstructed *values* at
all — it wants the raw 2/3-bit **indices** plus the per-entry scalars
(`vnorm`, `rnorm`, sign bits), because the level lookup is folded into
the precomputed products. So `kv_unpack`'s streaming output is indices,
and the level rescale stays in `kv_dequant` for the verification path.
