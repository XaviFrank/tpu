#!/usr/bin/env python3
"""Generate real RMSNorm test vectors for tb_rmsnorm.

TWO SITES, BOTH DRIVING THE SAME rtl/rmsnorm.sv AT DIFFERENT N:

  hidden   N=1024, layer-0 input_layernorm. x is a real token
           embedding, gamma the real learned input_layernorm weight.

  qknorm   N=128, layer-0 self_attn.q_norm. x is one real attention
           head's PRE-norm q vector (projection output, before RoPE),
           gamma the real learned q_norm weight.

The second site exists to demonstrate that rmsnorm.sv is already
generic in N: QK-norm is the same math over head_dim instead of the
hidden size, so it needs no new RTL - only a different instantiation
with its own eps_fixed and gamma, exactly as the two hidden-size sites
already differ from one another.

Golden for both is the numpy rmsnorm() in software/attention_l0.py,
already verified in the attention work - not re-derived here.

Emits into tb/, suffixed per site:
    rmsnorm_x_<site>.memh       N x INT16   quantized input
    rmsnorm_gamma_<site>.memh   N x Q2.14   signed, per-channel
    rmsnorm_exp_<site>.memh     N x INT8    expected output
    rmsnorm_params_<site>.memh  eps_fixed, scale_mult, shift_amt

FORMATS - see rmsnorm.sv's header for why each is per-site
    x        INT16, LSB s_x = calib_absmax / 32767
    gamma    Q2.14 SIGNED (the post_attention site's gamma reaches -1.492)
    eps      eps_float / s_x^2, i.e. already in mean_square units
    output   INT8 at out_scale = output_absmax / 127
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

from fetch_qwen import CACHE, fetch_embed_rows, fetch_file, load_header  # noqa: E402
import attention_l0 as A                                                # noqa: E402

OUT_FRAC = 24
QMAX8 = 127

SITES = {
    # site   -> (N,    gamma tensor,                            GAMMA_F, label)
    # GAMMA_F is PER SITE. Q2.14 covers the hidden sites (absmax 1.05
    # and 1.49) but NOT QK-norm: q_norm reaches 11.81 and k_norm 96.5,
    # which Q2.14 would silently saturate at 2.0. rmsnorm.sv already
    # parameterizes GAMMA_F, so this costs no RTL change.
    "hidden": (1024, "model.layers.0.input_layernorm.weight", 14, "input_layernorm"),
    "qknorm": (128,  "model.layers.0.self_attn.q_norm.weight", 11, "q_norm over head_dim"),
}


def site_vectors(site, seqs, rows, eps, cfg):
    """Return (all_calibration_vectors, the_one_under_test), float."""
    emb_per_seq = [np.stack([rows[int(t)] for t in s]).astype(np.float64)
                   for s in seqs]
    if site == "hidden":
        allv = np.concatenate(emb_per_seq)
        return allv, rows[int(seqs[0][-1])].astype(np.float64)

    # qknorm: exactly what forward_layer0 feeds q_norm - the q
    # projection output reshaped to heads, before RoPE.
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    g_in = A.T("model.layers.0.input_layernorm.weight")
    Wq = A.T("model.layers.0.self_attn.q_proj.weight")
    allv = []
    for x in emb_per_seq:
        h = A.rmsnorm(x, g_in, eps)
        allv.append((h @ Wq.T).reshape(-1, hd))
    allv = np.concatenate(allv)
    return allv, allv[0]


def main():
    site = sys.argv[1] if len(sys.argv) > 1 else "hidden"
    assert site in SITES, f"unknown site {site}; pick from {list(SITES)}"
    N, gamma_name, gamma_f_bits, label = SITES[site]

    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]
    gamma_f = A.T(gamma_name)
    assert gamma_f.shape == (N,), (gamma_f.shape, N)

    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    seqs = [tok.encode(t).ids for t in CALIB_TEXTS]
    ids = [i for s in seqs for i in s]
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    rows = fetch_embed_rows(hdr, ds, ids)

    allv, x_f = site_vectors(site, seqs, rows, eps, cfg)

    # scales from the CALIBRATION SET, not the single vector under test
    x_absmax = float(np.abs(allv).max())
    s_x = x_absmax / 32767.0
    out_absmax = float(np.abs(A.rmsnorm(allv, gamma_f, eps)).max())
    out_scale = out_absmax / QMAX8

    x_q = np.clip(np.rint(x_f / s_x), -32768, 32767).astype(np.int64)
    g_q = np.clip(np.rint(gamma_f * (1 << gamma_f_bits)), -32768, 32767).astype(np.int64)
    assert np.abs(gamma_f).max() * (1 << gamma_f_bits) <= 32767, (
        "gamma saturates at Q%d.%d - widen the integer part" % (16-gamma_f_bits, gamma_f_bits))
    eps_fixed = int(round(eps / (s_x * s_x)))

    denom = (1 << OUT_FRAC) * out_scale
    shift_amt = 0
    for sh in range(31, -1, -1):
        if round((1 << sh) / denom) <= 32767:
            shift_amt = sh
            break
    scale_mult = int(round((1 << shift_amt) / denom))

    out_f = A.rmsnorm(x_f, gamma_f, eps)
    exp_q = np.clip(np.rint(out_f / out_scale), -128, QMAX8).astype(np.int64)

    def w(name, vals, width):
        mask = (1 << width) - 1
        with open(os.path.join(HERE, name), "w") as f:
            f.write(chr(10).join("%0*x" % (width // 4, int(v) & mask)
                                 for v in vals) + chr(10))

    w("rmsnorm_x_%s.memh" % site, x_q, 16)
    w("rmsnorm_gamma_%s.memh" % site, g_q, 16)
    w("rmsnorm_exp_%s.memh" % site, exp_q, 8)
    w("rmsnorm_params_%s.memh" % site, [eps_fixed, scale_mult, shift_amt, gamma_f_bits], 48)

    print("site             %s (%s), N = %d" % (site, label, N))
    print("x        INT16   absmax %d  (s_x = %.6e)" % (np.abs(x_q).max(), s_x))
    print("gamma    Q%d.%d  min %d max %d" % (16-gamma_f_bits, gamma_f_bits, g_q.min(), g_q.max()))
    print("eps_fixed        %d" % eps_fixed)
    print("out_scale        %.8e  (from calibrated absmax %.5f)" % (out_scale, out_absmax))
    print("requant          scale_mult %d, shift_amt %d" % (scale_mult, shift_amt))
    print("expected INT8    min %d max %d, %d saturated"
          % (exp_q.min(), exp_q.max(), int((np.abs(exp_q) == QMAX8).sum())))


if __name__ == "__main__":
    main()
