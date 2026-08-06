#!/usr/bin/env python3
"""Per-channel mixed precision for K, matching the paper's stated setup.

TurboQuant reports "in our 2.5-bit setup, 32 outlier channels are
quantized at 3 bits, while the remaining 96 channels use 2 bits,
leading to an effective bit precision of (32*3+96*2)/128 = 2.5". Every
format this project has measured used UNIFORM allocation, which is a
real documented divergence from the paper and the cheapest specific
thing left to test.

A PREREQUISITE THE RESULT HINGES ON. The rotation exists precisely to
destroy per-channel structure - that is what "flattening outliers"
means. So before allocating bits per channel, this measures whether
outlier channels still EXIST in the rotated domain. If the per-
coordinate variance is flat, mixed precision cannot help by
construction, and that is itself the finding: the paper's outlier
channels must live somewhere this scheme has already rotated away.

Outlier channels are selected by per-coordinate variance of the
rotated, normalized K over the whole calibration population, which is
the only selection that is meaningful for a FIXED rotation - a
systematic component of K maps to systematically the same rotated
coordinates.

Same 24-row harness, same V-only control (0.1412), analysis only.
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
from gen_kv_codebooks import lloyd_max                        # noqa
from gen_kv_quant_vectors import codebook                     # noqa

NOUT = 32          # "outlier" channels, per the paper


def fit(z, bits):
    thr, lev = lloyd_max(z, 1 << bits)
    return (np.clip(np.rint(thr * 32768), -32768, 32767).astype(np.int64),
            np.clip(np.rint(lev * 32768), -32768, 32767).astype(np.int64))


def mixed_recon(yk, assign, books, qjl):
    """Quantize each coordinate with the book its channel is assigned."""
    sos = int((yk.astype(object) ** 2).sum())
    vn = math.isqrt(sos)
    rec = np.empty(N, dtype=np.int64)
    for i in range(N):
        thr, lev = books[assign[i]]
        t = [(int(x) * vn) >> 15 for x in thr]
        c = sum(1 for x in t if int(yk[i]) >= x)
        rec[i] = (int(lev[c]) * vn) >> 15
    if not qjl:
        return rec
    r = yk - rec
    rn = math.isqrt(int((r.astype(object) ** 2).sum()))
    amp = (rn * 2896) >> 15
    return rec + np.where(r < 0, -amp, amp)


def evaluate(rows, h, d, assign, books, qjl, bookV):
    thrV, levV = bookV
    lg, oe = [], []
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
        l_fmt = np.array([float(yq @ mixed_recon(h @ (d * kq[t]),
                                                 assign, books, qjl)) / 2**16
                          for t in range(T)])
        p_t, p_f = S.softmax2(l_true), S.softmax2(l_fmt)
        vh = np.stack([(h @ S.recon(h @ (d * vq[t]), thrV, levV, 3, 0))
                       * d / N * sv for t in range(T)])
        o_t = p_t @ (vq * sv)
        lg.append(np.sqrt(((l_fmt - l_true) ** 2).mean()))
        oe.append(np.linalg.norm(p_f @ vh - o_t) / np.linalg.norm(o_t))
    return np.mean(lg), np.mean(oe)


def main():
    h = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d = np.where(sign == 1, -1, 1)
    rows = S.rows()
    bookV = codebook(0, 3)

    # rotated, normalized K over the whole population
    Z = []
    for hd, qf, kf, vf in rows:
        sk = float(np.abs(kf).max()) / 127.0
        kq = np.clip(np.rint(kf / sk), -128, 127).astype(np.int64)
        for t in range(kq.shape[0]):
            y = (h @ (d * kq[t])).astype(np.float64)
            n = np.linalg.norm(y)
            if n > 0:
                Z.append(y / n)
    Z = np.stack(Z)

    var = Z.var(axis=0)
    order = np.argsort(-var)
    top, rest = order[:NOUT], order[NOUT:]
    print("PREREQUISITE - do outlier channels survive the rotation?")
    print("  per-coordinate variance  min %.3e  max %.3e  max/min %.2f"
          % (var.min(), var.max(), var.max() / var.min()))
    print("  mean var, top %d channels   %.4e" % (NOUT, var[top].mean()))
    print("  mean var, other %d channels %.4e" % (N - NOUT, var[rest].mean()))
    print("  ratio %.3f   (1.00 = perfectly flat, no outlier structure)"
          % (var[top].mean() / var[rest].mean()))
    print("")

    assign = np.zeros(N, dtype=int)
    assign[top] = 1

    # books fitted on the coordinates each group actually contains
    zl = Z[:, rest].reshape(-1)
    zh = Z[:, top].reshape(-1)
    b2l, b3l = fit(zl, 2), fit(zl, 3)
    b3h, b5h = fit(zh, 3), fit(zh, 5)
    uni2, uni3 = fit(Z.reshape(-1), 2), fit(Z.reshape(-1), 3)

    cfgs = [
        ("uniform 2b + QJL (deployed-equiv)", np.zeros(N, int), [uni2, uni2], 1, 2, 2),
        ("uniform 3b (no QJL)",               np.zeros(N, int), [uni3, uni3], 0, 3, 3),
        ("MIXED 32@3b / 96@2b  (paper 2.5b)", assign, [b2l, b3h], 0, 2, 3),
        ("MIXED 32@3b / 96@2b  + QJL",        assign, [b2l, b3h], 1, 2, 3),
        ("MIXED 32@5b / 96@3b  (paper 3.5b)", assign, [b3l, b5h], 0, 3, 5),
    ]

    print("%-36s %5s %5s  %-10s %-10s" %
          ("K format", "words", "b/co", "logit RMS", "OUTPUT rel"))
    for name, asg, bks, qjl, blo, bhi in cfgs:
        lg, oe = evaluate(rows, h, d, asg, bks, qjl, bookV)
        nbits = (N - NOUT) * blo + NOUT * bhi if asg.any() else N * blo
        words = -(-nbits // 32) + 1 + (5 if qjl else 0)
        print("%-36s %5d %5.2f  %-10.4f %-10.4f"
              % (name, words, words * 32.0 / N, lg, oe))
    print("")
    print("V-only control (exact logits, 3-bit V): 0.1412")
    print("true logit spread (sd across positions): 3.580 base-2 units")


if __name__ == "__main__":
    main()
