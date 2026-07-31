# software/ — Qwen3-0.6B quantization + packing pipeline

Python tooling, deliberately kept out of the hardware source tree.
Nothing here touches `rtl/`, `tb/`, `constraints/`, `regress.sh` or the
Vivado project. `regress.sh` is unchanged and still covers hardware only.

**v1 scope: a single-tensor proof.** Layer 0's `self_attn.q_proj`, one
16×16 tile of it, end to end — download → quantize → pack → golden.
Everything else (all 196 matmul tensors, RMSNorm gamma, embeddings /
LM head) is explicitly deferred.

**This pipeline runs entirely offline.** It has no board access and
does not talk to hardware. Its job ends at the artifact; running that
artifact on the board happens separately in the notebook.

## Layout

```
fetch_qwen.py       range-fetch the needed tensors from HF safetensors
quantize_pack.py    calibrate, quantize, pack, golden, emit artifact
qwen/               download cache (gitignored, ~19 MB)
out/                the deliverable (committed, ~15 KB)
```

Run in order:

```
python fetch_qwen.py
python quantize_pack.py
```

Dependencies: `numpy` and `tokenizers`. **No torch, no transformers** —
see "Why no torch" below.

## The deliverable

`out/qproj_l0_tile0.json` — self-describing, with `weight_words` (16 ×
128-bit, each with its four 32-bit `poke_lanes` ready to write),
`activation_words`, the scales, and `expected.golden_int32`.

Also `out/{weights_int8_loaded,activation_int8,golden_int32}.npy` for
anything that would rather have arrays than JSON.

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

**Per-tensor scaling costs ~6.5×, and that is the right next step.**
The weight scale comes from the full tensor (|max| 0.645) while this
tile only reaches 0.289, so the slice uses **57 of 127** int8 levels.
Per-output-channel scaling would cut error from 2.99% to 0.46% of span.
Not applied here on purpose: per-tensor is what a real deployment of
the whole tensor uses, and a slice-local scale would flatter the proof.

## Why no torch

`fetch_qwen.py` reads the safetensors header and pulls only the byte
ranges it needs — about 4 MB out of a 1.2 GB file — and converts
bfloat16 by shifting into float32. That keeps the whole pipeline on
`numpy` + `tokenizers` instead of a multi-GB install.

**The limit this hits:** layers 1–27 need a real forward pass to know
their activations, which this cannot do. Extending calibration past
layer 0 means installing torch/transformers or implementing the
transformer block in numpy. Layer 0 is exact precisely because nothing
precedes it.

## Deferred (not oversights)

- **All 196 tensors / 28 layers.** Also note a full Q-proj is
  [2048, 1024] = 64 K-tiles × 128 N-tiles, which needs
  `ws_tpu_tiled_top`; the current bitstream is single-tile
  `ws_tpu_top`, hence one 16×16 tile here.
- **Per-channel weight scales** — quantified above, ~6.5× better.
- **Calibration beyond layer 0** — needs a forward pass.
- **RMSNorm gamma** — elementwise, never touches the array; own spec.
- **Embeddings / LM head** — a gather, not a matmul. Note
  `tie_word_embeddings: true`, so there is no separate LM-head tensor.
