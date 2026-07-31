#!/usr/bin/env python3
"""Real layer-0 attention forward pass in numpy, to produce gate_proj's
genuine input activations.

WHY THIS EXISTS
  Q/K/V at layer 0 were free: their input is
  input_layernorm(embed_tokens[t]) and nothing else. gate_proj is not -
  its input is post_attention_layernorm(embedding + attention_output),
  so getting it right means computing the whole attention block:
  QK-norm, RoPE, GQA, causal softmax, O-proj, residual.

  This pass STOPS at those activations. No quantizing, no packing, no
  hardware artifact - that is a deliberate follow-up.

EVERY CONVENTION HERE WAS READ OUT OF THE REAL HF SOURCE, not recalled.
  qwen/modeling_qwen3.py (fetched from transformers main) and
  qwen/config.json are the references; line numbers below are that file.

    RMSNorm            (59-64)   float32, x*rsqrt(mean(x^2)+eps), then *weight
    QK-norm ordering   (254-256) q_norm(q_proj(h).view(...,head_dim)) THEN RoPE.
                                 Norm is over head_dim only, and V is NOT normed.
    rotate_half        (142-146) HALF-SPLIT: cat(-x[64:], x[:64]).
                                 NOT the interleaved-pairs variant.
    apply_rotary       (168-172) q*cos + rotate_half(q)*sin
    inv_freq           (123)     1/(theta**(arange(0,128,2)/128)), 64 entries
    cos/sin            (135-137) emb = cat(freqs, freqs) -> 128 wide
    repeat_kv          (175-184) repeat_INTERLEAVE on the head axis, so
                                 q head j uses kv head j // n_rep.
                                 The tile variant (j % n_kv) is WRONG and
                                 would be silently plausible - see check G.
    scaling            (223)     head_dim ** -0.5
    residual           (306-324) resid = x; x = attn(input_layernorm(x));
                                 x = resid + x; gate_proj input =
                                 post_attention_layernorm(x)

  config.json: rope_theta 1e6, attention_bias False (no biases anywhere),
  sliding_window None / use_sliding_window False (plain causal),
  rope_scaling None (so RoPE attention_scaling is 1.0).
"""

import json
import os

import numpy as np

from fetch_qwen import CACHE, fetch_embed_rows, fetch_file, load_header

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# A real sentence, same register as the existing calibration texts.
# Multi-token on purpose: a single token can only attend to itself, so
# causal masking, cross-token weighting and softmax would all be
# untested (spec section 1).
TEST_TEXT = "Systolic arrays exploit data reuse by streaming operands through a mesh."

L = "model.layers.0."


def T(name):
    return np.load(os.path.join(CACHE, name + ".npy")).astype(np.float64)


def rmsnorm(x, gamma, eps):
    """modeling_qwen3.py:59-64. Normalises over the LAST axis."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * gamma


def rotate_half(x):
    """modeling_qwen3.py:142-146 - half-split, not interleaved pairs."""
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]
    H = cfg["hidden_size"]
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    n_rep = nq // nkv
    theta = cfg["rope_theta"]
    assert cfg["attention_bias"] is False
    assert cfg["sliding_window"] is None and not cfg["use_sliding_window"]
    assert cfg["rope_scaling"] is None          # => RoPE attention_scaling = 1.0

    checks = {}

    # ---- 1. tokens + embeddings ---------------------------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    ids = tok.encode(TEST_TEXT).ids
    Tn = len(ids)
    assert 8 <= Tn <= 24, f"sequence length {Tn} outside the intended 8-16ish"
    header, dstart = load_header()
    header.pop("__metadata__", None)
    rows = fetch_embed_rows(header, dstart, ids)
    x = np.stack([rows[int(t)] for t in ids]).astype(np.float64)   # [T, H]
    print(f"sequence: {Tn} tokens -> {[tok.decode([i]) for i in ids]}")

    # ---- 2. input_layernorm --------------------------------------
    h = rmsnorm(x, T(L + "input_layernorm.weight"), eps)

    # ---- 3. Q/K/V projections (full matrices, no slicing) --------
    # nn.Linear stores [out, in] and computes x @ W.T
    q = h @ T(L + "self_attn.q_proj.weight").T          # [T, 2048]
    k = h @ T(L + "self_attn.k_proj.weight").T          # [T, 1024]
    v = h @ T(L + "self_attn.v_proj.weight").T          # [T, 1024]
    q = q.reshape(Tn, nq, hd)
    k = k.reshape(Tn, nkv, hd)
    v = v.reshape(Tn, nkv, hd)

    # ---- 4. QK-norm, over head_dim, BEFORE RoPE ------------------
    q = rmsnorm(q, T(L + "self_attn.q_norm.weight"), eps)
    k = rmsnorm(k, T(L + "self_attn.k_norm.weight"), eps)

    # ---- 5. RoPE -------------------------------------------------
    inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    pos = np.arange(Tn, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)                       # [T, 64]
    emb = np.concatenate([freqs, freqs], axis=-1)         # [T, 128]
    cos = np.cos(emb)[:, None, :]                         # [T,1,128] broadcast over heads
    sin = np.sin(emb)[:, None, :]
    q_pre_norm = np.linalg.norm(q, axis=-1)
    k_pre_norm = np.linalg.norm(k, axis=-1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    # CHECK R: rotation preserves L2 norm per head-vector.
    dq = np.abs(np.linalg.norm(q, axis=-1) - q_pre_norm).max()
    dk = np.abs(np.linalg.norm(k, axis=-1) - k_pre_norm).max()
    checks["rope_norm_preserved_q"] = float(dq)
    checks["rope_norm_preserved_k"] = float(dk)
    assert dq < 1e-9 and dk < 1e-9, f"RoPE changed norms: q {dq}, k {dk}"

    # ---- 6. GQA expansion: repeat_INTERLEAVE ---------------------
    # modeling_qwen3.py:175-184. np.repeat on the head axis is exactly
    # torch.repeat_interleave, so q head j pairs with kv head j//n_rep.
    k_e = np.repeat(k, n_rep, axis=1)                     # [T, 16, 128]
    v_e = np.repeat(v, n_rep, axis=1)

    # CHECK G: spot-check the mapping against the convention, and prove
    # the WRONG (tile) convention would actually differ - otherwise the
    # check is vacuous on this data.
    map_ok = all(np.array_equal(k_e[:, j, :], k[:, j // n_rep, :]) for j in range(nq))
    tiled = np.concatenate([k] * n_rep, axis=1)           # the wrong convention
    tile_differs = not np.array_equal(k_e, tiled)
    checks["gqa_head_map_matches_repeat_interleave"] = bool(map_ok)
    checks["gqa_tile_convention_would_differ"] = bool(tile_differs)
    assert map_ok, "GQA head mapping does not match repeat_interleave"
    assert tile_differs, "tile vs interleave indistinguishable here - check is vacuous"

    # ---- 7. scaled dot-product attention, causal -----------------
    scaling = hd ** -0.5
    qh = q.transpose(1, 0, 2)                             # [heads, T, hd]
    kh = k_e.transpose(1, 0, 2)
    vh = v_e.transpose(1, 0, 2)
    scores = np.einsum("htd,hsd->hts", qh, kh) * scaling  # [heads, T(query), T(key)]
    causal = np.triu(np.ones((Tn, Tn), dtype=bool), k=1)  # True above the diagonal
    scores = np.where(causal[None, :, :], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    w = e / e.sum(axis=-1, keepdims=True)                 # softmax

    # CHECK S: every query row sums to 1.
    row_err = float(np.abs(w.sum(axis=-1) - 1.0).max())
    checks["softmax_row_sum_max_err"] = row_err
    assert row_err < 1e-12, f"softmax rows do not sum to 1 (max err {row_err})"

    # CHECK C: attention to future positions is EXACTLY zero.
    future = w[:, causal]
    future_max = float(np.abs(future).max()) if future.size else 0.0
    checks["causal_future_weight_max"] = future_max
    checks["causal_future_exactly_zero"] = bool(np.all(future == 0.0))
    assert np.all(future == 0.0), f"future attention not exactly 0 (max {future_max})"
    # and the mask must be non-trivial, i.e. the sequence really is multi-token
    assert future.size > 0, "no masked positions - sequence is degenerate"

    attn = np.einsum("hts,hsd->htd", w, vh)               # [heads, T, hd]

    # CHECK V: each output is a convex combination of V rows (weights
    # are non-negative and sum to 1), so every component must lie within
    # the per-head, per-dimension min/max of V. Catches sign errors, a
    # transposed einsum, or weights applied to the wrong axis - all of
    # which produce plausible-looking numbers otherwise.
    vmin = vh.min(axis=1, keepdims=True)
    vmax = vh.max(axis=1, keepdims=True)
    tol = 1e-9
    within = bool(np.all(attn >= vmin - tol) and np.all(attn <= vmax + tol))
    checks["attn_within_convex_hull_of_v"] = within
    assert within, "attention output escapes the convex hull of V"

    # Sharpness, recorded so a future degenerate sequence is visible
    # rather than silently weakening the test.
    peak = w.max(axis=-1)
    ent = -(w * np.log(np.where(w > 0, w, 1.0))).sum(axis=-1)   # log(1)=0 on masked
    checks["attn_peak_weight_mean"] = float(peak.mean())
    checks["attn_frac_rows_over_0p99"] = float((peak > 0.99).mean())
    checks["attn_entropy_mean_nats"] = float(ent.mean())
    checks["attn_entropy_uniform_bound_nats"] = float(np.log(Tn))

    attn = attn.transpose(1, 0, 2).reshape(Tn, nq * hd)   # [T, 2048]

    # ---- 8. O-proj ------------------------------------------------
    o = attn @ T(L + "self_attn.o_proj.weight").T         # [T, 1024]

    # ---- 9. residual, against the PRE-norm input ------------------
    xr = x + o

    # ---- 10. post_attention_layernorm = gate_proj's input ---------
    gate_in = rmsnorm(xr, T(L + "post_attention_layernorm.weight"), eps)
    assert gate_in.shape == (Tn, H)

    # ---- report ---------------------------------------------------
    print("\nself-checks (spec section 4):")
    print(f"  RoPE norm preservation   q max delta {dq:.3e}, k max delta {dk:.3e}  -> PASS")
    print(f"  softmax row sums         max |sum-1| = {row_err:.3e}                 -> PASS")
    print(f"  causal future weights    exactly zero over {future.size} masked entries -> PASS")
    print(f"  GQA head mapping         matches repeat_interleave (j -> j//{n_rep})  -> PASS")
    print(f"    and the tile convention would have differed, so the check bites")
    print(f"  attn within hull of V    every component inside per-dim min/max     -> PASS")
    print("")
    print("  attention is non-degenerate: peak weight mean "
          "%.3f, %.1f%% of rows >0.99, entropy %.3f nats "
          "(uniform bound %.3f) - real cross-token mixing" % (
              checks['attn_peak_weight_mean'],
              100*checks['attn_frac_rows_over_0p99'],
              checks['attn_entropy_mean_nats'], np.log(Tn)))

    print("\nintermediate magnitudes (sanity, not assertions):")
    for nm, arr in [("embedding x", x), ("input_layernorm", h), ("q post-RoPE", q),
                    ("k post-RoPE", k), ("v", v), ("attn out", attn),
                    ("o_proj", o), ("residual", xr), ("GATE_PROJ INPUT", gate_in)]:
        print(f"  {nm:18s} shape {str(arr.shape):16s} absmax {np.abs(arr).max():10.5f} "
              f"std {arr.std():9.5f}")

    np.save(os.path.join(OUT, "gate_proj_input_l0.npy"), gate_in.astype(np.float32))
    meta = {
        "text": TEST_TEXT,
        "token_ids": [int(i) for i in ids],
        "tokens": [tok.decode([int(i)]) for i in ids],
        "seq_len": Tn,
        "hidden_size": H,
        "definition": "post_attention_layernorm(embed_tokens[t] + o_proj(attn(...)))"
                      " - the real input to layer 0 mlp.gate_proj / up_proj",
        "self_checks": checks,
        "conventions_verified_against": "huggingface transformers "
                                        "modeling_qwen3.py (fetched), config.json",
        "gate_proj_input_absmax": float(np.abs(gate_in).max()),
        "gate_proj_input_std": float(gate_in.std()),
        "per_token_absmax": [float(np.abs(gate_in[t]).max()) for t in range(Tn)],
        "note": "activations only - NOT quantized, NOT packed, no hardware "
                "artifact; that is the follow-up pass",
    }
    with open(os.path.join(OUT, "gate_proj_input_l0.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved out/gate_proj_input_l0.npy  [{Tn}, {H}] float32")
    print(f"      out/gate_proj_input_l0.json  (metadata + self-check results)")


if __name__ == "__main__":
    main()
