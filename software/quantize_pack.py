#!/usr/bin/env python3
"""Quantize, pack and golden-reference one real Qwen3-0.6B tensor slice.

Single-tensor proof: layer 0 self_attn.q_proj. Produces the artifact a
notebook can poke onto the board, plus the golden INT32 the hardware
must reproduce.

WHAT SLICE, AND WHY
  The current bitstream is ws_tpu_top - SINGLE TILE, N=K=16. One go
  computes a 16x16 matvec. Full Q-proj is [2048 out, 1024 in], i.e.
  64 K-tiles x 128 N-tiles, which needs the tiling engine
  (ws_tpu_tiled_top, not built into this bitstream). So this proof uses
  one real 16x16 corner of the real tensor: input dims 0..15, output
  dims 0..15. Real weights, real activations, real arithmetic - just
  one tile of it. Scaling out is the tiling engine's job, not this
  pipeline's.

ACTIVATION - REAL, AND EXACTLY COMPUTABLE
  Layer 0's q_proj input is input_layernorm(embed_tokens[token]), and
  nothing else. No attention, no MLP, no earlier layer feeds it. So the
  activation is closed-form from two tensors we already have, and needs
  no forward pass and no torch:
      x = embed_tokens[t]
      a = x * rsqrt(mean(x^2) + eps) * gamma
  That makes the calibration in this pass genuinely measured rather
  than heuristic, which is the whole point of spec section 3.

  THIS DOES NOT GENERALISE. Layers 1..27 need a real forward pass to
  know their inputs. Extending calibration past layer 0 means either
  installing torch/transformers or implementing the block in numpy -
  flagged in README.md, deliberately out of scope here.

PACKING - see spec section 4, answered from the RTL
  ws_controller reads the weight buffer DESCENDING:
      load_row  = (K-1) - load_cnt
      rd_addr_w = tile*K + load_row
  so the bottom-row-first shift order is produced by the controller's
  read sequence, NOT by the layout. The buffer address space is plain:
  word (tile*K + r) holds row r. The packer therefore does NOT reverse
  anything.

  The array computes y = W_loaded^T . a, so a host that wants y = W.a
  stores the tile TRANSPOSED:
      buffer word i, byte j  =  W_pt[j][i]
  with i the input/hidden index and j the output index, byte 0 in bits
  [7:0]. Both facts are asserted below by unpacking the words back and
  recomputing - a transpose or reversal bug produces a wrong answer
  silently otherwise.
"""

import json
import os
import sys

import numpy as np

from fetch_qwen import (CACHE, fetch_embed_rows, fetch_file, load_header)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

N = 16            # array size / output dims per tile
K = 16            # rows per weight tile / input dims per tile
QMAX = 127        # symmetric int8; -128 excluded so |q| is symmetric

# A small, deliberately varied calibration set - prose, code, digits,
# punctuation. Small on purpose (spec section 3: lightweight, not a
# calibration framework), but real text through the real tokenizer.
CALIB_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, Apollo 11 landed on the Moon; the crew returned six days later.",
    "def matmul(a, b):\n    return [[sum(x*y for x, y in zip(r, c)) for c in zip(*b)] for r in a]",
    "Systolic arrays exploit data reuse by streaming operands through a mesh of processing elements.",
    "Quantization maps floating-point tensors onto a small integer grid, trading precision for throughput.",
]


# Which tensor to pack. Q/K/V at layer 0 all read the SAME activation -
# there is one input_layernorm, and Qwen3's q_norm/k_norm are [head_dim]
# applied AFTER the projection - so calibration carries over unchanged.
TENSORS = {
    "q_proj": "model.layers.0.self_attn.q_proj.weight",
    "k_proj": "model.layers.0.self_attn.k_proj.weight",
    "gate_proj": "model.layers.0.mlp.gate_proj.weight",
}

# Where each tensor's input activation comes from. Q/K/V read
# input_layernorm(embed) directly. gate_proj does NOT - its input is
# post_attention_layernorm(embed + attention_out), so it needs the full
# verified attention block, and its own a_scale: measured absmax is
# 10.03 post-attention vs 3.19 pre-attention, so reusing Q/K/V's scale
# would waste most of the int8 range.
ACT_SOURCE = {
    "q_proj": "attn_in",
    "k_proj": "attn_in",
    "gate_proj": "mlp_in",
}


def rmsnorm(x, gamma, eps):
    """Qwen3 RMSNorm, computed in float32 exactly as the reference does."""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps)) * gamma).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "q_proj"
    assert which in TENSORS, f"unknown tensor {which}; pick from {list(TENSORS)}"
    tensor_name = TENSORS[which]
    print(f"tensor: {tensor_name}")

    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]
    hidden = cfg["hidden_size"]

    W_pt = np.load(os.path.join(CACHE, tensor_name + ".npy"))
    gamma = np.load(os.path.join(CACHE, "model.layers.0.input_layernorm.weight.npy"))
    # Shape is NOT hard-coded: k_proj is [1024,1024] where q_proj is
    # [2048,1024] (GQA has fewer KV heads). Only the input dim must match.
    assert W_pt.ndim == 2 and W_pt.shape[1] == hidden, W_pt.shape
    assert W_pt.shape[0] % N == 0 and W_pt.shape[1] % K == 0,         f"{W_pt.shape} does not tile cleanly at N={N}, K={K}"

    # ---- calibration: real tokens -> real layer-0 activations --------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    seqs = [tok.encode(t).ids for t in CALIB_TEXTS]
    all_ids = [i for s in seqs for i in s]
    header, data_start = load_header()
    header.pop("__metadata__", None)
    rows = fetch_embed_rows(header, data_start, all_ids)

    if ACT_SOURCE[which] == "attn_in":
        acts = np.stack([rmsnorm(rows[int(t)], gamma, eps) for t in all_ids])
    else:
        # full attention per sequence - position matters here, so this
        # cannot be done per unique token the way attn_in can be
        from attention_l0 import forward_layer0
        per_seq = [forward_layer0(s)[0] for s in seqs]
        acts = np.concatenate(per_seq, axis=0)
    a_absmax = float(np.abs(acts).max())
    a_p999 = float(np.percentile(np.abs(acts), 99.9))

    print(f"calibration: {len(seqs)} sequences, {len(all_ids)} tokens, "
          f"{len(set(all_ids))} unique")
    print(f"  activation |max| = {a_absmax:.6f}   99.9th pct = {a_p999:.6f}")

    # The sample vector under test: last token of the first sequence -
    # a real in-context token, chosen deterministically so the artifact
    # is reproducible.
    sample_tok = int(seqs[0][-1])
    if ACT_SOURCE[which] == "attn_in":
        a_full = rmsnorm(rows[sample_tok], gamma, eps)
    else:
        # same rule - last position of the first sequence - but the value
        # depends on the whole preceding context, not just the token id
        a_full = per_seq[0][-1]
    a_slice = a_full[:K]
    print(f"  sample token id  = {sample_tok} "
          f"({tok.decode([sample_tok])!r}), slice |max| = "
          f"{np.abs(a_slice).max():.6f}")

    # ---- scales --------------------------------------------------
    # Weight scale from the FULL tensor (that is what a real deployment
    # would use - a per-slice scale would flatter this proof).
    w_absmax = float(np.abs(W_pt).max())
    w_scale = w_absmax / QMAX
    # Activation scale from the measured calibration max.
    a_scale = a_absmax / QMAX
    print(f"  w_scale = {w_scale:.8e}  (from full-tensor |max| {w_absmax:.6f})")
    print(f"  a_scale = {a_scale:.8e}  (from calibration |max|)")

    # ---- quantize the slice --------------------------------------
    W_slice = W_pt[:N, :K]                       # [out 0..15, in 0..15]
    Wq_pt = np.clip(np.rint(W_slice / w_scale), -QMAX, QMAX).astype(np.int32)
    Aq = np.clip(np.rint(a_slice / a_scale), -QMAX, QMAX).astype(np.int32)

    # ---- golden INT32, straight from the quantized values ---------
    golden = (Wq_pt.astype(np.int64) @ Aq.astype(np.int64)).astype(np.int64)
    assert np.abs(golden).max() < 2**31, "int32 accumulator overflow"
    # headroom sanity: K * 127 * 127 is the theoretical bound
    print(f"  golden int32 range = [{golden.min()}, {golden.max()}] "
          f"(bound {K*QMAX*QMAX})")

    # ---- pack: word i = row i, byte j = W_pt[j][i] (transposed) ----
    W_loaded = Wq_pt.T                            # [in][out] = [row][col]
    weight_words = []
    for i in range(K):
        w = 0
        for j in range(N):
            w |= (int(W_loaded[i, j]) & 0xFF) << (8 * j)
        weight_words.append(w)
    act_word = 0
    for i in range(K):
        act_word |= (int(Aq[i]) & 0xFF) << (8 * i)

    # ---- self-check: unpack and recompute -------------------------
    # Catches a transpose/reversal bug in the packer, which would
    # otherwise produce a plausible-looking wrong answer on hardware.
    def s8(b):
        return b - 256 if b >= 128 else b

    unpacked = np.array([[s8((weight_words[i] >> (8 * j)) & 0xFF)
                          for j in range(N)] for i in range(K)], dtype=np.int64)
    a_unpacked = np.array([s8((act_word >> (8 * i)) & 0xFF)
                           for i in range(K)], dtype=np.int64)
    assert np.array_equal(unpacked, W_loaded.astype(np.int64)), "weight pack round-trip"
    assert np.array_equal(a_unpacked, Aq.astype(np.int64)), "activation pack round-trip"
    # y[j] = sum_i W_loaded[i][j] * a[i]  - the hardware's own convention
    recomputed = np.array([sum(int(unpacked[i, j]) * int(a_unpacked[i])
                               for i in range(K)) for j in range(N)], dtype=np.int64)
    assert np.array_equal(recomputed, golden), "packed layout does not reproduce golden"
    print("  pack/unpack round-trip and layout self-check: OK")

    # ---- float reference + error analysis -------------------------
    # NOTE ON METRICS. Per-element relative error is misleading here: a
    # 16-term dot product cancels, so some columns land near zero and
    # their relative error explodes while the ABSOLUTE error stays flat.
    # The meaningful figures are absolute error and error normalised by
    # the output range, so both are reported and carried in the artifact.
    y_ref = (W_slice.astype(np.float64) @ a_slice.astype(np.float64))
    y_deq = golden.astype(np.float64) * w_scale * a_scale
    abs_err = np.abs(y_deq - y_ref)
    span = float(np.abs(y_ref).max())
    rel = abs_err / np.maximum(np.abs(y_ref), 1e-12)
    print(f"  abs err        : max {abs_err.max():.6f}  mean {abs_err.mean():.6f}")
    print(f"  vs output span : max {abs_err.max()/span:.3%}  mean {abs_err.mean()/span:.3%}"
          f"   (span |y|max = {span:.5f})")
    print(f"  per-element rel: max {rel.max():.1%}  mean {rel.mean():.1%}"
          f"   <- inflated by near-zero columns, see note")

    # How much precision the deployment-realistic scale costs on this
    # tile, and what per-output-channel scaling would buy. Reported, not
    # applied: the artifact keeps the per-tensor scale because that is
    # what a real deployment of the whole tensor would use, and a
    # slice-local scale would flatter this proof.
    slice_absmax = float(np.abs(W_slice).max())
    levels_used = int(round(slice_absmax / w_scale))
    w_scale_pc = np.abs(W_slice).max(axis=1) / QMAX          # per output row
    Wq_pc = np.clip(np.rint(W_slice / w_scale_pc[:, None]), -QMAX, QMAX)
    y_pc = (Wq_pc.astype(np.float64) @ Aq.astype(np.float64)) * w_scale_pc * a_scale
    abs_err_pc = np.abs(y_pc - y_ref)
    print(f"  scale headroom : slice |max| {slice_absmax:.4f} vs tensor "
          f"{w_absmax:.4f} -> only {levels_used}/{QMAX} int8 levels used")
    print(f"  per-channel would give: max {abs_err_pc.max()/span:.3%} "
          f"mean {abs_err_pc.mean()/span:.3%} of span "
          f"({abs_err.max()/max(abs_err_pc.max(),1e-30):.1f}x better) - "
          f"the obvious next step when scaling out")

    # ---- poke lane values (32-bit each, lane 3 commits) ------------
    def lanes(word):
        return [(word >> (32 * l)) & 0xFFFFFFFF for l in range(4)]

    artifact = {
        "source": {
            "repo": "Qwen/Qwen3-0.6B",
            "tensor": tensor_name,
            "tensor_shape": list(W_pt.shape),
            "slice": {"out_dims": [0, N], "in_dims": [0, K],
                      "note": "one 16x16 tile of the real tensor; the "
                              "bitstream is single-tile ws_tpu_top"},
            "activation": {
                "definition": ("input_layernorm(embed_tokens[token])"
                               " - layer 0 only")
                              if ACT_SOURCE[which] == "attn_in" else
                              ("post_attention_layernorm(embed + o_proj(attn(...)))"
                               " - real layer-0 attention output, see attention_l0.py"),
                "source": ACT_SOURCE[which],
                "position": "last token of calibration sequence 0",
                "token_id": sample_tok,
                "token_text": tok.decode([sample_tok]),
            },
        },
        "quantization": {
            "scheme": "symmetric per-tensor int8, clip to +/-127",
            "w_scale": w_scale,
            "w_absmax_full_tensor": w_absmax,
            "a_scale": a_scale,
            "a_absmax_calibration": a_absmax,
            "a_p999_calibration": a_p999,
            "calibration_sequences": len(seqs),
            "calibration_tokens": len(all_ids),
            "dequant_factor": w_scale * a_scale,
            "dequant_note": "float_result = int32_from_hardware * dequant_factor",
        },
        "layout": {
            "rule": "buffer word i = row i (natural order); byte j = column j, byte 0 in bits[7:0]",
            "transpose": "word i, byte j = W_pt[j][i]  (array computes y = W_loaded^T . a)",
            "row_order_note": "ws_controller reads DESCENDING (load_row = K-1-load_cnt), "
                              "so bottom-first feeding is the controller's doing - "
                              "the packer must NOT reverse rows",
        },
        "weight_words": [
            {"addr": i, "hex": f"0x{weight_words[i]:032x}",
             "poke_lanes": lanes(weight_words[i])}
            for i in range(K)
        ],
        "activation_words": [
            {"addr": 0, "hex": f"0x{act_word:032x}", "poke_lanes": lanes(act_word)}
        ],
        "expected": {
            "golden_int32": [int(v) for v in golden],
            "check": "hardware PEEK buf_sel=3 must equal golden_int32 EXACTLY - "
                     "both sides are int32 arithmetic over identical int8 inputs, "
                     "so this is a bit-exact comparison, not an approximate one",
            "float_reference": [float(v) for v in y_ref],
            "dequantized": [float(v) for v in y_deq],
        },
        "error_analysis": {
            "note": "quantization error lives in the FLOAT comparison only; the "
                    "int32 check above is exact. Per-element relative error is "
                    "inflated by dot-product cancellation on near-zero columns - "
                    "use the span-normalised figures.",
            "abs_err_max": float(abs_err.max()),
            "abs_err_mean": float(abs_err.mean()),
            "output_span_absmax": span,
            "err_vs_span_max": float(abs_err.max() / span),
            "err_vs_span_mean": float(abs_err.mean() / span),
            "per_element_rel_max": float(rel.max()),
            "int8_levels_used_by_slice": levels_used,
            "per_channel_err_vs_span_max": float(abs_err_pc.max() / span),
        },
        "n_vecs": 1,
    }

    stem = which.replace("_proj", "proj") + "_l0_tile0"
    with open(os.path.join(OUT, stem + ".json"), "w") as f:
        json.dump(artifact, f, indent=2)
    np.save(os.path.join(OUT, stem + "_weights_int8_loaded.npy"), W_loaded.astype(np.int8))
    np.save(os.path.join(OUT, stem + "_activation_int8.npy"), Aq.astype(np.int8))
    np.save(os.path.join(OUT, stem + "_golden_int32.npy"), golden.astype(np.int32))

    print("")
    print("artifact -> " + os.path.join(OUT, stem + '.json'))
    print(f"golden int32: {[int(v) for v in golden]}")


if __name__ == "__main__":
    main()
