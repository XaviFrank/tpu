#!/usr/bin/env python3
"""Generate real RMSNorm test vectors for tb_rmsnorm's full-pipeline check.

Uses the ACTUAL Qwen3-0.6B layer-0 site-1 RMSNorm - x is a real token
embedding, gamma is the real learned input_layernorm weight - and the
golden output comes from the numpy rmsnorm() in software/attention_l0.py,
which is already verified (spec section 4: reuse it, do not re-derive
the math for the TB).

Emits, all into tb/:
    rmsnorm_x.memh       1024 x INT16   quantized residual-stream input
    rmsnorm_gamma.memh   1024 x Q2.14   signed, per-channel
    rmsnorm_exp.memh     1024 x INT8    expected output
    rmsnorm_params.memh  eps_fixed, scale_mult, shift_amt (48-bit hex)

FORMATS - see rmsnorm.sv's header for why each is per-site
    x        INT16, LSB s_x = calib_absmax / 32767
    gamma    Q2.14 SIGNED (site 2's gamma reaches -1.492)
    eps      eps_float / s_x^2, i.e. already in mean_square units
    output   INT8 at out_scale = output_absmax / 127, which the software
             pipeline already measured as q/k/v's a_absmax
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

N = 1024
OUT_FRAC = 24
QMAX8 = 127


def main():
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]

    gamma_f = A.T("model.layers.0.input_layernorm.weight")

    # real tokens -> real embeddings, exactly as the calibration does
    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    seqs = [tok.encode(t).ids for t in CALIB_TEXTS]
    ids = [i for s in seqs for i in s]
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    rows = fetch_embed_rows(hdr, ds, ids)
    emb = np.stack([rows[int(t)] for t in ids]).astype(np.float64)

    # scales from the CALIBRATION SET, not from the one vector under
    # test - a per-vector scale would flatter the result
    x_absmax = float(np.abs(emb).max())
    s_x = x_absmax / 32767.0
    out_absmax = float(np.abs(A.rmsnorm(emb, gamma_f, eps)).max())
    out_scale = out_absmax / QMAX8

    # the vector under test: last token of the first sequence, the same
    # deterministic pick the packing artifacts use
    x_f = rows[int(seqs[0][-1])].astype(np.float64)

    x_q = np.clip(np.rint(x_f / s_x), -32768, 32767).astype(np.int64)
    g_q = np.clip(np.rint(gamma_f * (1 << 14)), -32768, 32767).astype(np.int64)
    eps_fixed = int(round(eps / (s_x * s_x)))

    # requant config: int8 = sat((acc*scale_mult + 2^(sh-1)) >> sh) with
    # acc = out_real * 2^OUT_FRAC, so scale_mult/2^sh = 1/(2^F * out_scale).
    # Pick the largest shift that keeps scale_mult inside 16 signed bits.
    denom = (1 << OUT_FRAC) * out_scale
    shift_amt = 0
    for sh in range(31, -1, -1):
        if round((1 << sh) / denom) <= 32767:
            shift_amt = sh
            break
    scale_mult = int(round((1 << shift_amt) / denom))

    # GOLDEN: the verified numpy implementation, on the FLOAT inputs
    out_f = A.rmsnorm(x_f, gamma_f, eps)
    exp_q = np.clip(np.rint(out_f / out_scale), -128, QMAX8).astype(np.int64)

    def w(name, vals, width):
        mask = (1 << width) - 1
        with open(os.path.join(HERE, name), "w") as f:
            f.write("\n".join(f"{int(v) & mask:0{width // 4}x}" for v in vals) + "\n")

    w("rmsnorm_x.memh", x_q, 16)
    w("rmsnorm_gamma.memh", g_q, 16)
    w("rmsnorm_exp.memh", exp_q, 8)
    # hex so the TB can $readmemh it like every other generated vector
    with open(os.path.join(HERE, "rmsnorm_params.memh"), "w") as f:
        vals = [eps_fixed, scale_mult, shift_amt]
        f.write(chr(10).join("%012x" % v for v in vals) + chr(10))

    print(f"x        INT16   absmax {np.abs(x_q).max()}  (s_x = {s_x:.6e})")
    print(f"gamma    Q2.14   min {g_q.min()} max {g_q.max()}")
    print(f"eps_fixed        {eps_fixed}")
    print(f"out_scale        {out_scale:.8e}  (from calibrated absmax {out_absmax:.5f})")
    print(f"requant          scale_mult {scale_mult}, shift_amt {shift_amt}")
    print(f"expected INT8    min {exp_q.min()} max {exp_q.max()}, "
          f"{int((np.abs(exp_q) == QMAX8).sum())} saturated")
    print(f"wrote 4 files into {HERE}")


if __name__ == "__main__":
    main()
