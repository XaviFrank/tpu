# software/ — Qwen3-0.6B quantization + packing pipeline

Python tooling, deliberately kept out of the hardware source tree.
Nothing here touches `rtl/`, `tb/`, `constraints/`, `regress.sh` or the
Vivado project. `regress.sh` is unchanged and still covers hardware only.

**Scope: single-tile proofs.** Layer 0's `self_attn.q_proj` and
`self_attn.k_proj`, one 16×16 tile of each, end to end — download →
quantize → pack → golden. Everything else (the other 194 matmul
tensors, RMSNorm gamma, embeddings / LM head) is explicitly deferred.

`q_proj` tile 0 has been **verified bit-exact on real hardware**.

**This pipeline runs entirely offline.** It has no board access and
does not talk to hardware. Its job ends at the artifact; running that
artifact on the board happens separately in the notebook.

## Layout

```
fetch_qwen.py       range-fetch the needed tensors from HF safetensors
quantize_pack.py    calibrate, quantize, pack, golden, emit artifact
attention_l0.py     real layer-0 attention forward -> gate_proj's input
qwen/               download cache (gitignored, ~19 MB)
out/                the deliverables (committed)
```

Run in order:

```
python fetch_qwen.py
python quantize_pack.py q_proj      # default
python quantize_pack.py k_proj
python quantize_pack.py gate_proj  # needs attention_l0 for its activation
python quantize_pack.py up_proj
python quantize_pack.py down_proj  # needs the full SwiGLU intermediate
```

One parameterized script handles both, rather than a copy per tensor —
running *identical code* over two different shapes is what shows the
packer isn't shape-specific.

Dependencies: `numpy` and `tokenizers`. **No torch, no transformers** —
see "Why no torch" below.

## The deliverable

`out/qproj_l0_tile0.json` and `out/kproj_l0_tile0.json` —
self-describing, with `weight_words` (16 × 128-bit, each with its four
32-bit `poke_lanes` ready to write), `activation_words`, the scales,
and `expected.golden_int32`.

Also `out/<stem>_{weights_int8_loaded,activation_int8,golden_int32}.npy`
for anything that would rather have arrays than JSON.

**The activation is identical across the two.** Q/K/V at layer 0 all
read the same `input_layernorm` output — Qwen3's `q_norm`/`k_norm` are
`[head_dim]` and apply *after* the projection. Verified, not assumed:
the activation word, `a_scale` and the int8 activation array are
bit-identical between the two artifacts. A board already holding the
Q-proj activation only needs the 16 weight words re-poked.

### Two different checks, do not conflate them

| check | tolerance |
|---|---|
| hardware `PEEK buf_sel=3` vs `expected.golden_int32` | **exact** |
| `golden_int32 × dequant_factor` vs `float_reference` | quantization error |

The first is **bit-exact**: both sides are int32 arithmetic over the
same int8 inputs, so any mismatch is a real bug, not rounding. The
second is where quantization error lives, and it is a property of the
quantization scheme, not of the hardware.

## Findings worth knowing

**§2 verified against the real file, not arithmetic.** The audit reads
the actual safetensors header: **196** matmul weight tensors, shapes
only 1024/2048/3072, and **zero** dims that are not multiples of 16. No
ragged-tile handling is needed for this model at N=16.

**§4 answered from the RTL — the packer must NOT reverse rows.**
`ws_controller.sv:120,188`:

```systemverilog
assign load_row  = {KW{1'b1}} - load_cnt;   // K-1, K-2, ..., 0
assign rd_addr_w = w_idx_q * K + load_row;
```

Bottom-row-first feeding is produced by the *controller reading the
buffer in descending address order*. The address space is plain: word
`tile*K + r` holds row `r`. `tb_poke_peek` demonstrates the same thing
empirically.

Separately, the array computes `y = W_loadedᵀ · a`, so a host wanting
`y = W·a` stores the tile **transposed**:

```
buffer word i, byte j = W_pt[j][i]        (i = input dim, j = output dim)
```

`quantize_pack.py` asserts both by unpacking its own words and
recomputing the golden from them — a transpose or reversal bug would
otherwise produce a plausible wrong answer with no error anywhere.

**The activation is real and exactly computable, without a forward
pass.** Layer 0's q_proj input is `input_layernorm(embed_tokens[t])`
and nothing else — no attention or MLP feeds it. So calibration here is
genuinely measured (5 sequences, 102 real tokens through the real BPE
tokenizer), not a heuristic.

**Error is normal INT8, despite an alarming-looking relative figure.**
Measured on the tile: absolute error max 0.0108 / mean 0.0042, which is
**3.0% max and 1.2% mean of the output span**. Per-element *relative*
error reads 70% max — but only on columns where the result is near zero
(−0.015, 0.011). A 16-term dot product cancels, so relative error
explodes there while absolute error stays flat. Use the span-normalised
figures; the artifact carries both plus a note.

### Five tensors, same code — layer 0's attention and FFN complete

| | q_proj | k_proj | gate_proj | up_proj | down_proj |
|---|---|---|---|---|---|
| shape | [2048,1024] | [1024,1024] | [3072,1024] | [3072,1024] | [1024,3072] |
| activation source | attn_in | attn_in | mlp_in | mlp_in | **ffn_in** |
| K covered by tile | 1/64 | 1/64 | 1/64 | 1/64 | **1/192** |
| int8 levels /127 | 57 | 51 | 96 | 35 | 42 |
| abs err max | 0.0108 | 0.0057 | 0.0110 | 0.0101 | **0.0021** |
| err / partial span | 2.99% | 1.25% | 1.22% | 2.08% | 12.76% |
| **err / full output** | 2.39% | 0.63% | 0.51% | 1.19% | **0.33%** |

**Read the last row, not the one above it.** A tile computes a partial
sum over K, and normalising by that partial's span overstates the error
when the tile is a small slice of the real dot product. `down_proj` is
the extreme: it covers 16 of 3072 (1/192), and its partial output is
2.6% of the true magnitude (0.0164 vs 0.636) — which makes it look like
the worst tensor at 12.76% when its **absolute** error is the smallest
of all five and it is the *best* against the real output at 0.33%.
Same class of trap as the per-element relative error, one level up.

The artifacts carry both, plus `k_total` and `k_tiles_for_full_output`,
so the distinction cannot be lost.
Two things worth noting. The **under-use of int8 range reproduces**
(57 and 51 of 127), so it is a property of per-tensor scaling at tile
0, not a quirk of one tensor. But the **per-channel win does not
generalise**: 6.5× on q_proj, only 1.7× on k_proj, and k_proj's
per-channel result is actually *worse* in absolute terms than
q_proj's. Per-channel helps most where row magnitudes are widely
spread; it is still the right direction, but "6.5× better" was a
one-tensor number and should not be extrapolated.

**gate_proj needed its own calibration, and the numbers show why.**
Its input is `post_attention_layernorm(embed + attn_out)`, computed
through the verified attention block in `attention_l0.py` — not the
`input_layernorm(embed)` shortcut Q/K/V use. Measured absmax is
**10.03 against Q/K/V's 3.19**, so `a_scale` is 3.14× larger. Reusing
the Q/K/V scale would have clipped nothing but wasted about two thirds
of the int8 range.

Two things fall out of the third data point:

- **The int8-range under-use is not universal.** gate_proj reaches
  **96 of 127** levels where q_proj managed 57 and k_proj 51, because
  its tile-0 absmax (0.295) sits close to its tensor absmax (0.391).
  So per-tensor scaling is only wasteful when tile 0 happens to be
  unrepresentative — worth measuring per tensor rather than assuming.
- **The relative-error story holds up.** gate_proj's per-element
  relative error is only **9.7%** max, against 70% and 398% for q/k.
  Same code, same metric — the difference is that gate_proj's outputs
  do not cancel to near-zero. That is direct evidence the earlier
  alarming figures were dot-product cancellation, not a defect.

**Why per-tensor anyway.** On q_proj the weight scale comes from the
full tensor (|max| 0.645) while tile 0 only reaches 0.289, so the slice
gets 57 of 127 levels. Per-channel is not applied here on purpose:
per-tensor is what a real deployment of the whole tensor uses, and a
slice-local scale would flatter the proof. The numbers above are
reported so the trade is visible, not hidden.

## Real attention forward (`attention_l0.py`)

Q/K/V were free — their input is `input_layernorm(embed_tokens[t])` and
nothing else. `gate_proj` is not: its input is
`post_attention_layernorm(embedding + attention_output)`, so producing
it means computing the whole attention block in numpy — QK-norm, RoPE,
GQA, causal softmax, O-proj, residual.

Output: `out/gate_proj_input_l0.npy` `[14, 1024]` float32 plus a
metadata/self-check JSON. **Activations only** — not quantized, not
packed, no hardware artifact; that is the follow-up.

**Every convention was read out of the real HF source, not recalled.**
`qwen/modeling_qwen3.py` is fetched from transformers `main` and cited
by line number in the module docstring. The ones that would have been
silently wrong from memory:

- **`repeat_kv` is `repeat_interleave`** (175–184), so Q head `j` uses
  KV head `j // n_rep` — *not* `j % n_kv`. Both are plausible; only one
  is right.
- **`rotate_half` is half-split** (142–146), `cat(-x[64:], x[:64])` —
  not the interleaved-pairs RoPE variant.
- **QK-norm applies after projection and before RoPE** (254–256), over
  `head_dim` only, and **V is not normed**.
- Residual is against the **pre-norm** input (306–324).

### Self-check results

| check | result |
|---|---|
| RoPE preserves L2 norm | q 7.1e-15, k 1.1e-13 |
| softmax rows sum to 1 | max err 2.2e-16 |
| future attention weights | **exactly** 0.0 over 1456 masked entries |
| GQA map == `repeat_interleave` | true, and the tile variant *would* have differed |
| attn output inside convex hull of V | true |

The GQA check deliberately also asserts the wrong convention would
produce different numbers — otherwise it would pass vacuously.

Two things worth knowing, both verified rather than assumed:

- **`k_norm` gamma really does reach 96.5** (3 of 128 channels above
  10), which is why post-RoPE `k` reaches ~468 against `q`'s ~31. That
  asymmetry is a genuine property of the learned weights.
- **The sequence is not degenerate.** 14 tokens, peak attention weight
  averaging 0.749 and entropy 0.66 nats against a 2.64 uniform bound —
  real cross-token mixing, so causal masking and softmax are actually
  exercised.

## SwiGLU and down_proj's real input

`down_proj` is the one tensor whose tile cannot be built from sliced
matmuls. Its input is `silu(gate_proj(x)) * up_proj(x)` — the full
3072-wide gated intermediate — so `forward_layer0_ffn()` computes both
**full** 1024→3072 projections, applies SiLU and the elementwise
product, and only then takes the 16 values tile 0 consumes.
`up_proj` by contrast shares `gate_proj`'s activation exactly and is
mechanically the same as it.

Two self-checks on the gate, both non-vacuous:

- **Sign identity (exact).** `sign(silu(x)) == sign(x)` for all x, so
  `sign(inter)` must equal `sign(g)·sign(u)` everywhere. An algebraic
  identity, not a tolerance — catches a flipped sign, swapped
  gate/up, or SiLU on the wrong operand.
- **Collapse where the gate is strongly negative**, against a *tight
  analytic* bound rather than a guessed constant: `|silu|` peaks near
  x = −1.278 and decreases monotonically as x → −∞, so for all
  `g ≤ t` the exact supremum is `|silu(t)|`. Measured 0.0681 against
  the bound 0.0719 at t = −4.

A note on the second one: **SiLU is not monotonic** — it dips to about
−0.278 near x = −1.278 — so monotonicity is the wrong property to lean
on. The sign identity is what actually holds, and the collapse bound
follows from the tail behaviour rather than from monotonicity.

The threshold was also empirically chosen: an initial t = −8 made the
check **vacuous** (gate min is −5.11), and the non-vacuity assert caught
it rather than letting it silently pass.

## Why no torch

`fetch_qwen.py` reads the safetensors header and pulls only the byte
ranges it needs — about 4 MB out of a 1.2 GB file — and converts
bfloat16 by shifting into float32. That keeps the whole pipeline on
`numpy` + `tokenizers` instead of a multi-GB install.

**The limit this hits:** layers 1–27 need a full forward pass to know
their activations. `attention_l0.py` now implements one layer's
attention in numpy, so the remaining gap for layer 1+ is the MLP
(SwiGLU) and then iterating the block — no longer a torch dependency,
just more work. Layer 0 needed none of it for Q/K/V because nothing
precedes it.

## Deferred (not oversights)

- **All 196 tensors / 28 layers.** Also note a full Q-proj is
  [2048, 1024] = 64 K-tiles × 128 N-tiles, which needs
  `ws_tpu_tiled_top`; the current bitstream is single-tile
  `ws_tpu_top`, hence one 16×16 tile here.
- **Per-channel weight scales** — quantified above; 6.5× on q_proj but
  only 1.7× on k_proj, so measure per tensor rather than assuming.
- **Calibration beyond layer 0** — needs SwiGLU plus block iteration;
  attention itself is now done in numpy.
- **RMSNorm gamma** — elementwise, never touches the array; own spec.
- **Embeddings / LM head** — a gather, not a matmul. Note
  `tie_word_embeddings: true`, so there is no separate LM-head tensor.
