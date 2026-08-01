#!/usr/bin/env python3
"""Generate real SwiGLU test vectors for tb_swiglu_gate.

Golden comes from forward_layer0_ffn()'s already-verified SiLU/gating
computation (spec section 4: reuse it, do not re-derive SiLU for the
TB). Inputs are real gate_proj / up_proj outputs from the same
pipeline, quantized back into the INT32 accumulator domain the array
would actually produce.

Emits into tb/:
    swiglu_gate.memh    N x INT32  raw gate_proj accumulator
    swiglu_up.memh      N x INT32  raw up_proj accumulator
    swiglu_exp.memh     N x INT8   expected output
    swiglu_params.memh  g_scale_mult, g_shift, prod_shift,
                        scale_mult, shift_amt

SAMPLING
  A deterministic stride across the whole calibration set, PLUS the
  extreme-|gate| and extreme-|up| elements appended explicitly. Without
  those the tails would likely never be exercised - the real gate
  distribution has only 158 of 313344 values beyond |4|.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

from fetch_qwen import CACHE, fetch_file  # noqa: E402
import attention_l0 as A                  # noqa: E402

N = 1024
SILU_F = 12
QMAX8 = 127


def main():
    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))

    G, U = [], []
    for t in CALIB_TEXTS:
        h, _ = A.forward_layer0(tok.encode(t).ids)
        G.append(h @ A.T(A.L_ + "mlp.gate_proj.weight").T)
        U.append(h @ A.T(A.L_ + "mlp.up_proj.weight").T)
    G = np.concatenate(G).ravel()
    U = np.concatenate(U).ravel()

    # deterministic spread, then force the extremes in
    stride = max(1, G.size // (N - 4))
    sel = list(range(0, G.size, stride))[: N - 4]
    sel += [int(np.argmax(G)), int(np.argmin(G)),
            int(np.argmax(U)), int(np.argmin(U))]
    sel = np.array(sel[:N])
    g_f, u_f = G[sel], U[sel]

    # accumulator scales, from the same per-tensor quantization the
    # packing artifacts use: s = w_scale * a_scale
    a_scale = float(np.abs(np.concatenate([A.forward_layer0(
        tok.encode(t).ids)[0] for t in CALIB_TEXTS])).max()) / QMAX8
    wg = float(np.abs(A.T(A.L_ + "mlp.gate_proj.weight")).max()) / QMAX8
    wu = float(np.abs(A.T(A.L_ + "mlp.up_proj.weight")).max()) / QMAX8
    s_g, s_u = wg * a_scale, wu * a_scale

    gate_acc = np.rint(g_f / s_g).astype(np.int64)
    up_acc = np.rint(u_f / s_u).astype(np.int64)

    # gate dequant: xq(Q4.12) = gate_acc * s_g * 2^SILU_F
    target = s_g * (1 << SILU_F)
    g_shift = 0
    for sh in range(31, -1, -1):
        if round(target * (1 << sh)) <= 32767:
            g_shift = sh
            break
    g_scale_mult = int(round(target * (1 << g_shift)))

    # keep the product inside requant's 32-bit input
    prod_max = int(np.abs(np.rint(g_f * (1 << SILU_F))).max()) * int(np.abs(up_acc).max())
    prod_shift = 0
    while prod_max >> prod_shift >= (1 << 30):
        prod_shift += 1

    # GOLDEN: the verified pipeline's own SiLU/gating, on FLOAT inputs
    gated = (g_f * A._sigmoid(g_f)) * u_f
    out_absmax = float(np.abs((G * A._sigmoid(G)) * U).max())
    out_scale = out_absmax / QMAX8
    exp_q = np.clip(np.rint(gated / out_scale), -128, QMAX8).astype(np.int64)

    # requant: int8 = rq_in * scale_mult >> shift_amt, and
    # rq_in = gated * 2^SILU_F / (s_u * 2^prod_shift)
    denom = (1 << SILU_F) / (s_u * (1 << prod_shift)) * out_scale
    shift_amt = 0
    for sh in range(31, -1, -1):
        if round((1 << sh) / denom) <= 32767:
            shift_amt = sh
            break
    scale_mult = int(round((1 << shift_amt) / denom))

    def w(name, vals, width):
        mask = (1 << width) - 1
        with open(os.path.join(HERE, name), "w") as f:
            f.write("\n".join("%0*x" % (width // 4, int(v) & mask) for v in vals) + "\n")

    w("swiglu_gate.memh", gate_acc, 32)
    w("swiglu_up.memh", up_acc, 32)
    w("swiglu_exp.memh", exp_q, 8)
    w("swiglu_params.memh",
      [g_scale_mult, g_shift, prod_shift, scale_mult, shift_amt], 32)

    print("samples          %d (extremes forced in)" % N)
    print("gate real        min %.4f max %.4f" % (g_f.min(), g_f.max()))
    print("gate_acc         min %d max %d" % (gate_acc.min(), gate_acc.max()))
    print("up_acc           min %d max %d" % (up_acc.min(), up_acc.max()))
    print("g dequant        scale_mult %d shift %d  (target %.6e)"
          % (g_scale_mult, g_shift, target))
    print("prod_shift       %d" % prod_shift)
    print("out_scale        %.8e (from gated absmax %.5f)" % (out_scale, out_absmax))
    print("requant          scale_mult %d shift_amt %d" % (scale_mult, shift_amt))
    print("expected INT8    min %d max %d, %d saturated"
          % (exp_q.min(), exp_q.max(), int((np.abs(exp_q) == QMAX8).sum())))


if __name__ == "__main__":
    main()
