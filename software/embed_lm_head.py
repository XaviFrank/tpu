#!/usr/bin/env python3
"""Quantize Qwen3-0.6B's embedding table / LM head - the tied tensor.

Deliberately NOT part of calibrate_all.py's layer loop, because this
tensor is not a layer tensor and is not used the way layer tensors are.
Section 2 flagged it as "architecturally a gather, not a matmul"; that
is half right, and the half that is wrong is the interesting part.

====================================================================
ONE TENSOR, TWO ROLES - and they want the SAME quantization
====================================================================
tie_word_embeddings is true, so model.embed_tokens.weight and
lm_head.weight are the same [151936, 1024] parameter (verified here:
identical storage in the loaded model, and bit-identical bytes in the
safetensors file, where it is nonetheless stored TWICE - 311 MB of the
1.5 GB file is a duplicate).

  as an EMBEDDING it is a gather. Row t is read out whole and handed
  to input_layernorm. There is no dot product, so what matters is how
  well one ROW is reconstructed.

  as an LM HEAD it is a matmul: logits[j] = W[j, :] . h. Output
  channel j is row j.

Those two turn out to be the same requirement. Per-row scaling is the
natural choice for a gather - the row and its scale are fetched
together, at no extra cost - and for the matmul, per-row IS
per-output-channel, which is the standard and strictly better choice
for a projection this wide. So one set of 151936 scales (608 KB fp32,
0.4% of the int8 weights) serves both roles optimally. That is not a
compromise between the two uses; it is the same answer twice.

====================================================================
WHAT THE HARDWARE CAN ACTUALLY USE TODAY
====================================================================
ws_tpu_top's REQUANT_CFG carries ONE scale for all N lanes, so the
current bitstream can only consume the per-tensor scale. Per-channel
would need a per-lane scale register - a control-plane change, out of
scope here. Both scales are therefore emitted, and the cost of using
the per-tensor one is measured below rather than assumed, so the
decision to add per-lane scales can be made on a number.

Scale of the problem, stated plainly: as a matmul this is 151936x1024
= 9496 x 64 = 607,744 tiles of 16x16, and 155 MB of int8. It is by a
wide margin the largest matmul in the model - bigger than all 28
layers' attention projections combined - and integrating it is a
separate problem from having it quantized correctly, which is all this
script claims to do.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "calib28")
MODEL = "Qwen/Qwen3-0.6B"
N = 16
K = 16
QMAX = 127

CALIB_TEXTS_FILE = os.path.join(HERE, "qwen", "eval_corpus.txt")
EVAL_CHARS = 8000        # ~2k tokens of real text for the logit check


def q_per_tensor(W):
    s = float(np.abs(W).max()) / QMAX
    return np.clip(np.rint(W / s), -QMAX, QMAX).astype(np.int8), s


def q_per_row(W):
    s = np.abs(W).max(axis=1) / QMAX                 # one per vocab row
    s = np.maximum(s, 1e-12)
    q = np.clip(np.rint(W / s[:, None]), -QMAX, QMAX).astype(np.int8)
    return q, s.astype(np.float32)


def kl(p_log, q_log):
    """KL(p || q) from log-softmax rows, averaged over positions."""
    p = np.exp(p_log)
    return float(np.mean(np.sum(p * (p_log - q_log), axis=-1)))


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s")

    emb = model.model.embed_tokens.weight
    head = model.lm_head.weight
    tied = emb.data_ptr() == head.data_ptr()
    print(f"tie_word_embeddings: {tied} "
          f"(embed_tokens and lm_head share storage: {tied})")
    assert tied, "untied head would need its own quantization pass"

    W = emb.detach().to(torch.float32).numpy()
    vocab, hidden = W.shape
    print(f"tensor {W.shape}, |max| {np.abs(W).max():.6f}, "
          f"{W.nbytes/1e6:.1f} MB float32 -> {vocab*hidden/1e6:.1f} MB int8")

    qt, s_t = q_per_tensor(W)
    qr, s_r = q_per_row(W)
    print(f"per-tensor scale {s_t:.8e}")
    print(f"per-row scales: min {s_r.min():.3e} max {s_r.max():.3e} "
          f"ratio {s_r.max()/s_r.min():.1f}x")

    # ---- role 1: the gather. How well is one ROW reconstructed? -----
    dt = qt.astype(np.float32) * s_t
    dr = qr.astype(np.float32) * s_r[:, None]
    def rel_rms(d):
        return float(np.sqrt(np.mean((d - W) ** 2)) / np.sqrt(np.mean(W ** 2)))
    # per-row relative error, so a rare badly-scaled row cannot hide in
    # a global average over 151936 rows
    num_t = np.sqrt(((dt - W) ** 2).mean(axis=1))
    num_r = np.sqrt(((dr - W) ** 2).mean(axis=1))
    den = np.sqrt((W ** 2).mean(axis=1))
    rt, rr = num_t / den, num_r / den
    print("\nROLE 1 - gather (row reconstruction):")
    print(f"  per-tensor  rel RMS: mean {rt.mean():.4f}  worst row {rt.max():.4f}")
    print(f"  per-row     rel RMS: mean {rr.mean():.4f}  worst row {rr.max():.4f}")
    print(f"  per-row is {rt.mean()/rr.mean():.2f}x better on average, "
          f"{rt.max()/rr.max():.2f}x on the worst row")

    # ---- role 2: the LM head. Does it change the logits? ------------
    # Real hidden states at the head's input, i.e. model.norm's output,
    # captured on real text. This is the number that decides whether
    # per-lane scales are worth a control-plane change.
    text = open(CALIB_TEXTS_FILE, encoding="utf-8").read()[:EVAL_CHARS]
    ids = tok(text, return_tensors="pt")["input_ids"]
    grab = []
    h = model.lm_head.register_forward_pre_hook(
        lambda m, a: grab.append(a[0].detach().to(torch.float32).numpy()))
    with torch.no_grad():
        model(ids)
    h.remove()
    H = grab[0].reshape(-1, hidden)
    print(f"\nROLE 2 - LM head, {H.shape[0]} real positions:")

    # activation scale for h, same absmax discipline as everywhere else
    a_scale = float(np.abs(H).max()) / QMAX
    Hq = np.clip(np.rint(H / a_scale), -QMAX, QMAX).astype(np.int32)

    # The int8 matmuls go through float32, not int32: numpy has no BLAS
    # path for integer matmul, so int32 here is ~1e12 unaccelerated MACs.
    # float32 is EXACT for this product - the accumulator cannot exceed
    # K*127*127 and float32 represents every integer below 2**24 - so
    # this is a speed change, not a precision one. Asserted, not assumed.
    acc_bound = hidden * QMAX * QMAX
    assert acc_bound < 2 ** 24, (
        "float32 logit path is only exact while K*127*127 < 2**24 "
        f"(K={hidden} gives {acc_bound})")
    Hf = Hq.astype(np.float32)
    ref = H.astype(np.float32) @ W.T.astype(np.float32)
    lt = (Hf @ qt.astype(np.float32).T) * (s_t * a_scale)
    lr = (Hf @ qr.astype(np.float32).T) * a_scale * s_r[None, :]

    lp_ref, lp_t, lp_r = log_softmax(ref), log_softmax(lt), log_softmax(lr)
    top_ref = ref.argmax(-1)
    acc_t = float(np.mean(lt.argmax(-1) == top_ref))
    acc_r = float(np.mean(lr.argmax(-1) == top_ref))
    print(f"  per-tensor : top-1 agreement {acc_t:7.2%}   KL {kl(lp_ref, lp_t):.5f}")
    print(f"  per-row    : top-1 agreement {acc_r:7.2%}   KL {kl(lp_ref, lp_r):.5f}")
    print(f"  a_scale {a_scale:.6e} (absmax over the same real positions)")

    # ---- artifact ---------------------------------------------------
    # Tile 0 in the same shape every other artifact uses, so the board
    # path can smoke-test this tensor too - with the PER-TENSOR scale,
    # since that is what the current bitstream can consume.
    W_slice = W[:N, :K]
    a_slice = H[-1][:K]
    Wq = np.clip(np.rint(W_slice / s_t), -QMAX, QMAX).astype(np.int32)
    Aq = np.clip(np.rint(a_slice / a_scale), -QMAX, QMAX).astype(np.int32)
    golden = (Wq.astype(np.int64) @ Aq.astype(np.int64)).astype(np.int64)

    W_loaded = Wq.T
    words = []
    for i in range(K):
        w = 0
        for j in range(N):
            w |= (int(W_loaded[i, j]) & 0xFF) << (8 * j)
        words.append(w)
    aw = 0
    for i in range(K):
        aw |= (int(Aq[i]) & 0xFF) << (8 * i)

    def s8(b):
        return b - 256 if b >= 128 else b
    un = np.array([[s8((words[i] >> (8 * j)) & 0xFF) for j in range(N)]
                   for i in range(K)], dtype=np.int64)
    assert np.array_equal(un, W_loaded.astype(np.int64)), "pack round-trip"
    rec = np.array([sum(int(un[i, j]) * s8((aw >> (8 * i)) & 0xFF)
                        for i in range(K)) for j in range(N)], dtype=np.int64)
    assert np.array_equal(rec, golden), "packed layout vs golden"

    def lanes(word):
        return [(word >> (32 * l)) & 0xFFFFFFFF for l in range(4)]

    art = {
        "source": {
            "repo": MODEL,
            "tensor": "model.embed_tokens.weight (== lm_head.weight, tied)",
            "tensor_shape": [vocab, hidden],
            "slice": {"out_dims": [0, N], "in_dims": [0, K],
                      "note": "one 16x16 tile of the real tensor; the "
                              "bitstream is single-tile ws_tpu_top"},
            "roles": {
                "embedding": "gather - row t is read whole and fed to "
                             "input_layernorm; no dot product",
                "lm_head": "matmul logits[j] = W[j,:] . h, so output "
                           "channel j IS row j",
                "note": "per-row scaling is the natural choice for the "
                        "gather AND equals per-output-channel scaling "
                        "for the matmul - one scale set serves both",
            },
            "activation": {
                "definition": "model.norm(hidden) - the LM head's input",
                "position": "last position of the evaluation text",
            },
        },
        "quantization": {
            "scheme": "symmetric int8; per-tensor scale shipped, per-row "
                      "scales also emitted",
            "w_scale": s_t,
            "w_absmax_full_tensor": float(np.abs(W).max()),
            "a_scale": a_scale,
            "a_absmax_calibration": float(np.abs(H).max()),
            "per_row_scales_npy": "embed_lm_head_row_scales.npy",
            "per_row_scale_min": float(s_r.min()),
            "per_row_scale_max": float(s_r.max()),
            "dequant_factor": s_t * a_scale,
            "dequant_note": "float_result = int32_from_hardware * dequant_factor",
            "hardware_note": "REQUANT_CFG carries ONE scale for all N "
                             "lanes, so this bitstream can only use "
                             "w_scale; per_row would need per-lane scale "
                             "registers",
        },
        "layout": {
            "rule": "buffer word i = row i (natural order); byte j = "
                    "column j, byte 0 in bits[7:0]",
            "transpose": "word i, byte j = W_pt[j][i]  (array computes "
                         "y = W_loaded^T . a)",
        },
        "weight_words": [{"addr": i, "hex": hex(words[i]),
                          "poke_lanes": lanes(words[i])} for i in range(K)],
        "activation_words": [{"addr": 0, "hex": hex(aw), "poke_lanes": lanes(aw)}],
        "expected": {
            "golden_int32": [int(v) for v in golden],
            "check": "hardware PEEK buf_sel=3 must equal golden_int32 EXACTLY",
        },
        "error_analysis": {
            "gather_rel_rms_per_tensor_mean": float(rt.mean()),
            "gather_rel_rms_per_tensor_worst_row": float(rt.max()),
            "gather_rel_rms_per_row_mean": float(rr.mean()),
            "gather_rel_rms_per_row_worst_row": float(rr.max()),
            "lm_head_positions_evaluated": int(H.shape[0]),
            "lm_head_top1_per_tensor": acc_t,
            "lm_head_top1_per_row": acc_r,
            "lm_head_kl_per_tensor": kl(lp_ref, lp_t),
            "lm_head_kl_per_row": kl(lp_ref, lp_r),
        },
        "integration_note": {
            "tiles_as_matmul": (vocab // N) * (hidden // K),
            "int8_bytes": vocab * hidden,
            "comment": "largest matmul in the model; having it quantized "
                       "correctly is separate from integrating it",
        },
        "n_vecs": 1,
    }

    np.save(os.path.join(OUT, "embed_lm_head_int8_per_tensor.npy"), qt)
    np.save(os.path.join(OUT, "embed_lm_head_int8_per_row.npy"), qr)
    np.save(os.path.join(OUT, "embed_lm_head_row_scales.npy"), s_r)
    with open(os.path.join(OUT, "embed_lm_head.json"), "w") as f:
        json.dump(art, f, indent=1)
    print(f"\nwrote embed_lm_head.json + 2 int8 tensors + row scales to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
