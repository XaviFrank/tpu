# KV cache quantized write path — design notes

Companion to `ws_design_notes.md`. Covers the write side only: rotate,
quantize, append. Attention compute (Q·Kᵀ, softmax, ·V) is a separate,
later spec and is **not** described here.

---

## 1. Which rotation does TurboQuant actually specify?

**Finding: a dense random orthogonal matrix. Not Hadamard.**

The paper is *TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate*, arXiv 2504.19874 — Zandieh, Daliri, Hadian, Mirrokni
(Google Research / NYU / Google DeepMind). Its methodology states:

> "We start by randomizing this vector by multiplying it with a random
> rotation matrix **Π ∈ ℝ^(d×d)**. We can generate Π by applying **QR
> decomposition on a random matrix with i.i.d Normal entries**."

The guarantee is proven *for that construction*. Everything downstream
rests on

> "The resulting rotated vector, **Π·x**, is uniformly distributed on
> the unit sphere **S^(d-1)**."

That uniformity is what induces the Beta coordinate distribution the
fixed Lloyd–Max scalar codebook is designed against. It is exactly what
a Haar-random orthogonal matrix gives and what a randomized Hadamard
transform gives only approximately.

**The words "Hadamard", "Walsh", "O(d log d)" and "FFT" do not appear
anywhere in the paper**, including implementation, experiments and
appendices. It nowhere acknowledges dense rotation as a cost, and
nowhere offers a faster alternative.

### Where the "TurboQuant uses Hadamard" claim comes from

There *is* a Hadamard variant — **TurboQuant-H** — but it is **third
party** (Karen Mosoyan, Henry Ndubuaku; Cactus Compute), not the paper's
authors. It describes the original as "Random orthogonal matrix via QR
of Gaussian, O(d²)" and substitutes "Normalized Hadamard matrix,
O(N log N), symmetric = self-inverse." Its stated justification
explicitly excludes our use case:

> "For **offline weight quantization** we don't need the data-oblivious
> guarantees of a random rotation... The Hadamard rotation still
> concentrates coordinates, which is all we need."

KV cache quantization is the *online, data-oblivious* case — the one
that sentence carves out. Several secondary write-ups conflate the two
and describe the original as using a fast Walsh–Hadamard transform.
That is wrong, and it is the reason this investigation was gated ahead
of any RTL.

---

## 2. What it costs here

Real dims from `software/qwen/config.json`: `head_dim` 128, 28 layers,
8 KV heads → **448 rotations per token**, d = 128.

| | Dense Π (paper) | Randomized Hadamard (HD) |
|---|---|---|
| ops / vector | 16,384 MAC | 896 add/sub, **0 mult** |
| ops / token | 7.34 MMAC | 401K add/sub |
| array time / token, 16×16 @215 MHz | ~29.7K cyc ≈ **138 µs** batched | negligible |
| new DSPs | 0 *if time-shared with `ws_array`*, ~256 dedicated | **0** |
| inverse | Πᵀ (transpose read) | self-inverse |

Π is **fixed and data-oblivious**, so the rotation is a fixed-matrix
matrix-vector product — precisely what the weight-stationary array
already does. It can reuse `ws_array` + `ws_tiling_engine`
(128×128 = 64 tiles of 16×16) at zero new DSPs, spending array time
instead. Batching the 16 K/V vectors per layer amortizes weight reload
to ~50% utilization (~267 µs/token, ~3% of an ~8 ms token); the Week-4
double-buffered prefetch could hide most of that.

So dense is *affordable* — it buys a scheduling and buffering problem,
not a free lunch.

**This inverts the acceptance check in the spec's §8.** Seeing DSPs is
no longer evidence of a mistake. For a dense-mode build, seeing *zero*
DSPs would mean the rotation never reached a multiplier.

---

## 3. Two things the paper never had to face, because it ran in float

1. **Fixed point erodes the dense advantage.** Π's entries are
   ~N(0, 1/128), sd ≈ 0.0884. Quantizing Π *itself* to INT8 perturbs
   the very orthogonality the 16K MACs are being spent on. A normalized
   Hadamard's ±1/√128 entries are **exact** in fixed point under a
   single shared scale.
2. **d = 128 is where the Hadamard approximation is weakest.**
   Randomized-Hadamard concentration is asymptotic in d with log
   factors, and so is the paper's exact-Beta assumption. At d = 128 the
   gap between HD and Haar-random is likelier to be visible than at
   d = 4096. This cuts both ways — it argues for measuring, not for
   assuming either construction.

### Prediction, recorded before the measurement

Stated up front so the step-5 round-trip number can confirm or refute
it rather than be rationalized after the fact:

- **Hadamard path error should be dominated purely by 3-bit
  quantization.** `H_unnorm` and the Rademacher `D` are exact integers,
  and `H_unnormᵀ·H_unnorm = 128·I` **exactly**, so the forward and
  inverse transforms are lossless in integer arithmetic. The only error
  is the quantizer.
- **Dense path error should be strictly larger**: 3-bit quantization
  *plus* a residual non-orthogonality term, because `Π_int8ᵀ·Π_int8 ≠ I`
  once Π is quantized to INT8.

**So the prediction is that the cheap transform wins on this hardware,
for a reason that does not exist on the GPU the paper was written for.**
If the measurement contradicts this, the modelling above is wrong and
the note gets corrected, not the number.

### Refinement after step 2 — the predicted mechanism is real but negligible

Measured, `tb/gen_kv_rot_tables.py`:

| | max off-diagonal | rms off-diagonal |
|---|---|---|
| Π in float | 9.99e-16 | — |
| **Π quantized to INT8** | **5.29e-03** | **1.23e-03** |
| Randomized Hadamard | 0 (exact) | 0 (exact) |

So the non-orthogonality predicted above is real and measurable. **But
it is ~150× smaller than the error it would have to compete with**: an
optimal 8-level scalar quantizer on a Gaussian source runs ~14.6 dB
SNR, i.e. relative rms error ≈ 0.19, against Π's 1.2e-3. The
orthogonality penalty will be *swamped* by 3-bit quantization noise and
is not going to decide anything.

**The prediction was directionally right and practically irrelevant.**
The honest correction: the mechanism that actually separates the two
modes is not orthogonality, it is *how well each flattens outliers*,
which step 2 measured directly (`tb_kv_rotate`, real layer-0 vectors,
hardware output):

| peak/rms | input | Hadamard out | dense Π out |
|---|---|---|---|
| K (post-k_norm, post-RoPE) | 10.911 | **1.658** | 3.137 |
| V (raw v_proj, not normed) | 6.305 | **2.546** | 2.815 |

Both flatten hard; Hadamard flattens *further* on both vectors.

### But flatter is not automatically better — a trap for step 3

The dense numbers (3.137, 2.815) are not a defect. A Haar rotation
makes coordinates ~Gaussian, and peak/rms ≈ 3 is simply what 128
Gaussian samples look like — it is the **Beta density the paper's
Lloyd–Max codebook is fitted to**. The Hadamard output is flatter than
Gaussian (1.658) because `H` spreads energy deterministically: a spike
input becomes all ±1, peak/rms exactly 1.

So the two modes produce **different output distributions**, and a
single codebook cannot be optimal for both. Scoring the Hadamard path
with the paper's Beta-fitted codebook would handicap it for a reason
that has nothing to do with the rotation, and would make the step-5
comparison invalid.

**Step 3 must therefore fit one codebook per `ROT_MODE`**, each to its
own mode's empirical output distribution, and say which is which.
Recorded here because it is exactly the kind of mismatch that produces
a confident, wrong conclusion.

---

## 3b. The step-1 question, answered — measured round trip, 816 vectors each

Full path, real layer-0 vectors, hardware-exact model:
`x → rotate → quantize → pack → append → read back → reconstruct →
inverse rotate → x_hat`, compared against the original `x`.

| | mean | sd | worst | best |
|---|---|---|---|---|
| Hadamard, V (3b) | 13.74 dB | 1.35 | 7.91 | 16.57 |
| dense Π, V (3b) | 13.51 dB | 1.21 | 8.18 | 15.35 |
| Hadamard, K (2b+QJL) | **15.72 dB** | **4.38** | 2.80 | 23.78 |
| dense Π, K (2b+QJL) | 10.15 dB | 1.90 | 1.98 | 12.34 |

**V is a tie.** Hadamard is +0.22 dB on the mean and *−0.27 dB* at the
worst case. Nothing separates them.

**K favours Hadamard on the mean by +5.57 dB — but the worst case is
+0.82 dB, essentially a tie, and Hadamard's spread is more than double
(sd 4.38 vs 1.90).** Hadamard is better on average and less
predictable. A plausible cause, *not verified here*: QJL reconstruction
spreads the residual norm evenly across coordinates, which fits the
Hadamard path's flatter residual better than the Haar path's Gaussian
one. Confirming that needs the read side.

Note this **inverts the step-3 rotated-domain reading**, where dense Π
had the tighter K/V spread (0.02 dB vs 2.87 dB). End to end the spread
is Hadamard 1.98 dB, dense Π 3.36 dB. The tight spread dense Π shows at
the quantizer does not survive the QJL stage and the inverse rotation —
which is exactly why the round trip, not the intermediate statistic, is
the thing the spec asked to measure.

### Recommendation

**Take the Hadamard path (`ROT_MODE = 0`) as the default.** It is free —
no DSP, no 16 KB ROM, negligible cycles against dense Π's ~3% of a
token — and across 3264 measured round trips it is never meaningfully
worse: +0.82 dB at K's worst case, 0.27 dB behind at V's.

This does **not** match what TurboQuant prescribes, and the reason is
specific to this hardware: the paper's guarantee buys predictable
distortion on unseen distributions, and in float on a GPU that costs
almost nothing extra. Here the dense rotation additionally has to be
quantized to INT8, at d=128, where its theoretical advantage is
measurably absent.

The guarantee argument is *weakened, not dead* — Hadamard's higher K
variance is the shape of exactly the concern it names, and only layer 0
has been characterized. The cheap insurance is that `ROT_MODE` stays a
parameter: switching costs a codebook re-fit and array time, not a
redesign.

## 4. Decisions taken

**Rotation — pluggable, both built, measured.** `ROT_MODE` selects
dense-Π or randomized-Hadamard behind one interface. The literature
question is settled; the question that actually governs the datapath is
whether the cheap transform loses accuracy *on this data, in INT8, at
d=128*, and that is measurable on hardware already in the repo. The two
paths live in `generate` branches so a Hadamard-configured synth truly
carries no multiplier and no 16 KB Π ROM.

**Storage format — asymmetric between K and V.** The paper's
inner-product quantizer stores three parts:

> `Q_prod(x) = [Q_mse(x), Q_qjl(x − Q_mse⁻¹(Q_mse(x))), ‖x − Q_mse⁻¹(Q_mse(x))‖₂]`

MSE-optimal quantizers are *biased for inner products* — that is the
paper's own stated reason for the QJL stage. Attention scores are inner
products. So:

- **K** → 2-bit MSE + 1-bit QJL residual + residual norm (unbiased for
  Q·Kᵀ).
- **V** → pure 3-bit MSE. V is consumed as a probability-weighted sum,
  where reconstruction error is what matters.

This matters because "write side only" does not actually defer the
question: the write path fixes the format the deferred read side must
live with. A pure-MSE format would also make the step-4 round-trip TB
*look* excellent — reconstruction error is the exact metric an MSE
quantizer optimizes — while being the wrong format for attention. That
is a vacuity trap of the same shape as the RMSNorm gamma and RoPE
threshold findings.

**Bit width.** The spec asks for 3 bits. The paper reports **3.5 bits**
(quality-neutral) and **2.5 bits** (marginal degradation), both via
mixed precision — e.g. 2.5 = 32 outlier channels @3b + 96 @2b. Uniform
3-bit is a sound engineering choice but is *not* a number from the
paper, and is recorded as a deviation.

**Cache residency.** BRAM-resident at a modest test size for now.
~2.35 GB at INT8 for the full 40,960-token context means DRAM residency
is unavoidable at real context lengths regardless of compression;
DRAM-backed addressing is real future work, deliberately not required
to prove the quantize/append mechanism.

---

## 5. Sources

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)
- [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2504.19874)
- [TurboQuant-H — Cactus Compute (third-party Hadamard variant)](https://docs.cactuscompute.com/v1.14/blog/turboquant-h/)
