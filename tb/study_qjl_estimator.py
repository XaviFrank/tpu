#!/usr/bin/env python3
"""Does TurboQuant's actual QJL estimator close the K-format gap?

The write path stores sign(r_i) per coordinate plus ||r||, and the read
path so far RECONSTRUCTED the residual as r_hat = (||r||/sqrt(N))*sign(r)
and took <q, r_hat>. That reconstruction has the correct NORM, which is
why it looked reasonable, but it is not an inner-product estimator.

The paper's Q_qjl is a 1-bit Quantized JL transform. For a random
projection s,

    E[ <s,q> * sign(<s,k>) ] = <q,k> * sqrt(2/pi) / ||k||

so the UNBIASED estimate is

    <q,k> ~ ||k|| * sqrt(pi/2) / m * <Sq, sign(Sk)>

Here the rotation H*D IS the projection and m = N, and the score is
carried in the rotated domain (where y_Q . y_K = N*(q.k)), so the
residual's contribution to the score is

    rnorm * sqrt(pi/2) / N * dotB       dotB = sum_i y_Q,i * sign(y_r,i)

against the deployed  rnorm / sqrt(N) * dotB. The deployed term is
sqrt(2N/pi) = 9.03x too large at N=128.

CRUCIALLY dotB IS ALREADY WHAT THE HARDWARE COMPUTES, and the stored
bits are unchanged - so if this closes the gap it is a read-side
constant, not a write-path format change.

Same 24-row harness and the same V-only control as study_k_format.py.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "software"))

import study_k_format as S                                    # noqa
from gen_kv_rot_tables import N, hadamard                     # noqa
from gen_kv_quant_vectors import codebook, quantize           # noqa


def run(rows, h, d, bookK, bookV, qjl_scale, bits=2):
    """qjl_scale multiplies rnorm*dotB. None disables the QJL term."""
    thrK, levK = bookK
    thrV, levV = bookV
    oe, le = [], []
    for hd, qf, kf, vf in rows:
        T = kf.shape[0]
        sk = float(np.abs(kf).max()) / 127.0
        sv = float(np.abs(vf).max()) / 127.0
        sq = float(np.abs(qf).max()) / 127.0
        kq = np.clip(np.rint(kf / sk), -128, 127).astype(np.int64)
        vq = np.clip(np.rint(vf / sv), -128, 127).astype(np.int64)
        qq = np.clip(np.rint(qf / sq), -128, 127).astype(np.int64)

        G = sq * sk * math.log2(math.e) * (1 << 16) / (N * math.sqrt(hd))
        yq = np.rint((h @ (d * qq)) * G).astype(np.int64)

        l_true = (qf @ kf.T) * math.log2(math.e) / math.sqrt(hd)
        l_fmt = []
        for t in range(T):
            yk = h @ (d * kq[t])
            _, q_, _, _, vn, rn = quantize(yk, thrK, levK, bits, 1)
            rec = np.array([(int(levK[c]) * vn) >> 15 for c in q_], dtype=np.int64)
            s = float(yq @ rec)
            if qjl_scale is not None:
                r = yk - rec
                dotB = float(np.sum(np.where(r < 0, -yq, yq)))
                s += qjl_scale * rn * dotB
            l_fmt.append(s / 2**16)
        l_fmt = np.array(l_fmt)

        p_t, p_f = S.softmax2(l_true), S.softmax2(l_fmt)
        vh = np.stack([(h @ S.recon(h @ (d * vq[t]), thrV, levV, 3, 0))
                       * d / N * sv for t in range(T)])
        o_t = p_t @ (vq * sv)
        oe.append(np.linalg.norm(p_f @ vh - o_t) / np.linalg.norm(o_t))
        le.append(np.sqrt(((l_fmt - l_true) ** 2).mean()))
    return np.mean(le), np.mean(oe)


def main():
    h = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d = np.where(sign == 1, -1, 1)
    rows = S.rows()
    bookK, bookV = codebook(0, 2), codebook(0, 3)

    dep = 1.0 / math.sqrt(N)                 # deployed reconstruction
    est = math.sqrt(math.pi / 2) / N         # paper's estimator

    print("24 real causal rows, K = deployed 2b book + stored QJL bits.")
    print("Nothing about the STORED format changes - only how the read")
    print("side interprets the sign bits.")
    print("")
    print("%-34s %-12s %-12s" % ("QJL term", "logit RMS", "OUTPUT rel"))
    for name, sc in (("none (MSE only)", None),
                     ("reconstruction  rnorm/sqrt(N)", dep),
                     ("ESTIMATOR  rnorm*sqrt(pi/2)/N", est)):
        lg, oo = run(rows, h, d, bookK, bookV, sc)
        print("%-34s %-12.4f %-12.4f" % (name, lg, oo))

    print("")
    print("scale sweep, to check the optimum sits where the theory says:")
    print("%-14s %-10s %-12s" % ("x estimator", "scale", "OUTPUT rel"))
    best = (None, 1e9)
    for mult in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 9.03):
        _, oo = run(rows, h, d, bookK, bookV, est * mult)
        tag = "  <- theory" if mult == 1.0 else (
              "  <- deployed" if abs(mult - 9.03) < 0.01 else "")
        print("%-14.2f %-10.6f %-12.4f%s" % (mult, est * mult, oo, tag))
        if oo < best[1]:
            best = (mult, oo)
    print("")
    print("best multiplier %.2f (output rel %.4f)" % best)
    print("V-only control (exact logits, 3-bit V): 0.1412")


if __name__ == "__main__":
    main()
