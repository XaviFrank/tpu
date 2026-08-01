#!/usr/bin/env python3
"""Generate real RoPE rotation vectors for tb_rope.

Input is a REAL layer-0 attention head's q vector taken exactly where
RoPE sits in the pipeline: after q_proj and q_norm, before the
attention product. Quantized to INT8 with the calibration-set scale,
the same treatment the hardware's upstream rmsnorm instance produces.

THE SCALE CANCELS. A rotation is linear and magnitude-preserving, so
    rotate(x * s) / s == rotate(x)
and the golden can be computed entirely in INT8 units with exact
cos/sin - no dequantize/requantize round trip, and no scale to get
wrong. That is also why rope.sv needs no requant tap.

    rot_x1 = round(x1*cos(p*theta_j) - x2*sin(p*theta_j))
    rot_x2 = round(x2*cos(p*theta_j) + x1*sin(p*theta_j))

with j the frequency index, x1 = x[j], x2 = x[j+64].

Emits into tb/:
    rope_x1.memh    64 x INT8   low half of the head vector
    rope_x2.memh    64 x INT8   high half
    rope_exp1.memh  64 x INT8   expected rotated low half
    rope_exp2.memh  64 x INT8   expected rotated high half
    rope_pos.memh   the position, 32-bit
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

from fetch_qwen import CACHE, fetch_embed_rows, fetch_file, load_header  # noqa: E402
import attention_l0 as A                                                # noqa: E402

POSITION = 137          # a real, arbitrary mid-sequence position
HEAD = 3                # which attention head to take


def main():
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]
    hd, base = cfg["head_dim"], cfg["rope_theta"]
    nq, nf = cfg["num_attention_heads"], cfg["head_dim"] // 2

    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    seqs = [tok.encode(t).ids for t in CALIB_TEXTS]
    ids = [i for s in seqs for i in s]
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    rows = fetch_embed_rows(hdr, ds, ids)

    g_in = A.T("model.layers.0.input_layernorm.weight")
    norms = {"q": A.T("model.layers.0.self_attn.q_norm.weight"),
             "k": A.T("model.layers.0.self_attn.k_norm.weight")}
    projs = {"q": A.T("model.layers.0.self_attn.q_proj.weight"),
             "k": A.T("model.layers.0.self_attn.k_proj.weight")}
    nheads = {"q": nq, "k": cfg["num_key_value_heads"]}

    # Both Q and K, each post its own norm - exactly what
    # forward_layer0 feeds into RoPE. The rotation itself is identical
    # for the two; only the preceding norm and the head count differ.
    for which in ("q", "k"):
        emit(which, cfg, eps, hd, base, nf, seqs, rows, g_in,
             norms[which], projs[which], nheads[which])


def emit(which, cfg, eps, hd, base, nf, seqs, rows, g_in, gnorm, Wp, nh):
    allq = []
    for sq in seqs:
        x = np.stack([rows[int(t)] for t in sq]).astype(np.float64)
        h = A.rmsnorm(x, g_in, eps)
        v = (h @ Wp.T).reshape(len(sq), nh, hd)
        allq.append(A.rmsnorm(v, gnorm, eps).reshape(-1, hd))
    allq = np.concatenate(allq)

    scale = float(np.abs(allq).max()) / 127.0
    vec = allq[HEAD]                     # one real head vector
    xq = np.clip(np.rint(vec / scale), -128, 127).astype(np.int64)

    x1, x2 = xq[:nf], xq[nf:]
    th = np.array([base ** (-2.0 * j / hd) for j in range(nf)])
    ang = POSITION * th
    c, s_ = np.cos(ang), np.sin(ang)

    e1 = np.clip(np.rint(x1 * c - x2 * s_), -128, 127).astype(np.int64)
    e2 = np.clip(np.rint(x2 * c + x1 * s_), -128, 127).astype(np.int64)

    sat = int(((np.abs(np.rint(x1 * c - x2 * s_)) > 127).sum()
               + (np.abs(np.rint(x2 * c + x1 * s_)) > 127).sum()))

    def w(name, vals, width):
        mask = (1 << width) - 1
        with open(os.path.join(HERE, name), "w") as f:
            f.write(chr(10).join("%0*x" % (width // 4, int(v) & mask)
                                 for v in vals) + chr(10))

    w("rope_x1_%s.memh" % which, x1, 8)
    w("rope_x2_%s.memh" % which, x2, 8)
    w("rope_exp1_%s.memh" % which, e1, 8)
    w("rope_exp2_%s.memh" % which, e2, 8)
    w("rope_pos.memh", [POSITION], 32)

    # norm preservation, in the exact arithmetic - this is the property
    # the hardware TB then re-measures with LUT cos/sin
    n_in = float(np.sum(x1.astype(float)**2 + x2.astype(float)**2))
    n_out = float(np.sum((x1*c - x2*s_)**2 + (x2*c + x1*s_)**2))
    print("--- %s ---" % which)
    print("position         %d, head %d" % (POSITION, HEAD))
    print("input  INT8      min %d max %d  (scale %.6e)" % (xq.min(), xq.max(), scale))
    print("expected INT8    x1 [%d, %d]  x2 [%d, %d]" % (e1.min(), e1.max(), e2.min(), e2.max()))
    print("saturated        %d of %d components" % (sat, 2*nf))
    print("exact-math norm  in %.4f  out %.4f  rel diff %.2e"
          % (math.sqrt(n_in), math.sqrt(n_out),
             abs(math.sqrt(n_out) - math.sqrt(n_in)) / math.sqrt(n_in)))


if __name__ == "__main__":
    main()
