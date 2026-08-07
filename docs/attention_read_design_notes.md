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

| entries | step | curvature-only rel. error | effective bits |
|---|---|---|---|
| 128 | 0.0078 | 3.67e-06 | 18.1 |
| **256** | **0.0039** | **9.16e-07** | **20.1** |
| 512 | 0.0020 | 2.29e-07 | 22.1 |

**256 intervals + 1 endpoint, 16-bit Q1.15** — 4.1 kbit, against RoPE's
262 kbit. Q1.15 rather than Q0.16 because `2^0 = 1.0` would be 65536
and overflow the word; in Q1.15 the range `(0.5, 1]` maps to
`(16384, 32768]`. 257 entries, not 256, for the same reason
`swiglu_gate` needed a 257th: linear interpolation of the last interval
needs the `f=1` endpoint, and `2^-f` is not periodic. (RoPE's
identical-looking wrap *is* correct, because sine is.)

### Correction: the table's accuracy is 13.5 bits, not 20.1

The 20.1 above is **curvature only** — the error of a linear
interpolant against a smooth function. It ignores the two things the
hardware actually does: round the table entries to Q1.15 integers, and
truncate the interpolation product by `>> 8`. Measured end-to-end
against exact `2^-f` (`tb/gen_attn_exp2.py`):

```
  worst abs 1.489 LSB    worst rel 8.588e-05    = 13.5 effective bits
```

So the limit is the **word width, not the table length** — the error is
integer LSB rounding in the interpolation arithmetic, and adding
entries would not move it. The lever, if it were ever needed, is a
wider word.

It is not needed: 13.5 bits is ~1e-4 relative, three orders of
magnitude below V's 3-bit quantization floor (~13 dB ≈ 2.2 bits). The
conclusion that the table is not the limiting term survives; the number
I predicted for it did not.

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

> **SUPERSEDED IN PART — see §5.** The K half of this section predates
> the end-to-end study that moved K from 2-bit to INT8; at 8 bits the K
> inner loop is a real per-coordinate MAC and the mux argument does not
> apply. The V half is still sound as *analysis* but was **never
> implemented**: `attn_scan.sv` multiplies per coordinate on the V side
> too. §5 has the measured site inventory, the LANES cost model, and
> the decision.

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

---

## §5 — The V-side level precompute: measured, costed, **not implemented**

Added after the first synthesis run of `attn_scan_gqa` returned **20
DSP48E2** at `NQ=2, LANES=1`. §3 above predicted "~8–12 DSPs total, and
if synthesis reports DSPs proportional to lane count, the
level-precompute factoring did not happen." Both halves of that
sentence need correcting, and the correction goes in opposite
directions.

### What §3 got wrong

**§3's K analysis is obsolete, not incorrect.** It was written when K
was 2-bit, where a 4:1 mux over `P[i][k] = yQ[i]·lev_q15[k]` really did
remove the multiplier. The end-to-end study moved K to INT8 (top-1
53.3% → 91.2%), and 256 levels per coordinate cannot be muxed. `kprod =
yq_i * k_s` is a genuine per-coordinate MAC and always will be. That is
a consequence of a measured accuracy decision, not a regression.

**§3's V analysis was never implemented.** `attn_scan.sv` computes
`pv = cw_s * lv_sel` inside `S_VACC`, once per coordinate. The 8:1 mux
over per-position precomputed products exists in the design note and in
the module header; it does not exist in the RTL. The header's claim
that "V stays 3-bit and stays mux-based, so the V loop is still
multiplier-free" is false as built, and is corrected in place.

### The actual site inventory

Four multiplies per `attn_scan` instance, by declared operand width:

| site | line | operands | scope | ~DSP48E2 (27×18) |
|---|---|---|---|---|
| `kprod = yq_i * k_s`   | 232 | 32 × 9 signed   | **per coordinate** | 2 |
| `eterm = ediff*ffrac`  | 297 | 18 × 8 signed   | per position | 1 |
| `cw = w_q * v_vnorm`   | 322 | 32 × 30 unsigned| per position | 4 |
| `pv = cw_s * lv_sel`   | 327 | 48 × 16 signed  | **per coordinate** | 2 |

Predicted 9/instance, 18 at `NQ=2`; measured **20**. The 2-DSP residual
is most likely conservative operand splitting on `cw` or `pv` and is
**not attributed** — settling it needs the hierarchical cell report,
which should split 10/10 across `gen_q[0].u` and `gen_q[1].u`.

Note `yq` is **32 bits**, not 8. The INT8 conversion widened it to hold
the larger folded gain once the `1/N` from the rotation went away. "K's
8×8 MAC" is a statement about the *format*; the *datapath* is 32×9.

### The tradeoff, and why LANES=1 changes the answer

Only the two per-coordinate sites replicate with lane count:

```
  as built          per instance = 4·LANES + 5      NQ=2:  8·LANES + 10
  with §3 precompute per instance = 2·LANES + 7     NQ=2:  4·LANES + 14
```

|            | LANES=1 | LANES=2 | LANES=4 | LANES=16 |
|---|---|---|---|---|
| as built, NQ=2       | 18 | 26 | 42 | **138** |
| §3 precompute, NQ=2  | 18 | 22 | 30 | **78**  |

**At LANES=1 the precompute saves exactly zero DSPs.** This is the part
the earlier "~154 vs ~80" framing obscured: those are LANES=16 numbers.
At one lane there is exactly one `pv` multiplier in the RTL, and it is
already time-shared across all 128 coordinates. The precompute replaces
128 uses of one multiplier with 8 uses of one multiplier — the *same
one multiplier*. It is a **time-multiplexing** optimization, and time
multiplexing only pays when the thing being multiplexed is replicated.

What it would cost at LANES=1, for that zero gain:

- 8 × 48-bit product registers (~384 FF)
- an 8:1 48-bit mux (~150 LUT)
- 8 extra cycles per position between `S_WEIGHT` and `S_VACC` — a **+6%
  per-position latency** on the 128-cycle V pass
- one more state in a verified sequencer

Strictly worse on every axis. Crossover is **LANES=2**.

### Decision

**Keep the per-coordinate multiply. Do not implement the precompute
now.** Revisit it as a *precondition* of raising LANES, not as an
independent optimization — at LANES=16 it is worth 60 DSPs across the
head pair, which is the difference between fitting alongside the 288-DSP
WS array and not.

When it is implemented, the serial 8-cycle form is wrong at high lane
counts for a throughput reason that is easy to miss: at LANES=16 the V
pass is only 8 cycles, so a serial precompute **doubles** it, and the
per-position budget (15.5 cycles/entry) is already tight. Use 4
multipliers × 2 cycles (8 DSPs, +2 cycles) rather than 1 × 8.

**No interface change.** `attn_scan`'s port list and timing contract are
untouched by this decision, so anything building against them is safe.

### A second, larger saving — also declined, for a different reason

`cw = w_q * v_vnorm` is 4 DSPs and the single biggest consumer per
instance at LANES=1. `w_q` is `eval15 << GUARD >> nsh`: only 16 bits of
mantissa plus a variable shift. Multiplying `eval15` (16b) by `v_vnorm`
(30b) *before* the shift is 2 DSPs instead of 4 — but it reorders
truncation, so it is **not bit-exact** and would invalidate every golden
in `tb_attn_scan`, `tb_attn_scan_gqa` and `tb_attn_e2e` plus the
accuracy study behind the format choice. Declined as a numerics change
masquerading as an area optimization. Recorded so it is not
rediscovered as free.
