#!/usr/bin/env python3
"""Expected packed cache lines for tb_kv_quantize.

Reads the step-2 rotation goldens back off disk rather than recomputing
them, so this stage is verified against exactly the bits the rotation
testbench already proved bit-exact. If step 2's goldens were wrong,
this does not paper over it.

EVERY OPERATION HERE IS THE RTL'S, IN INTEGERS. The scaled threshold is
(t_q15 * vnorm) >> 15 with arithmetic shift - Python's >> on negative
ints already floors, matching a Verilog >>> on a signed value - and the
norms are floor(sqrt(.)) from a restoring integer square root. There is
no float anywhere in the expected values, so the testbench compares
bit-exact rather than with a tolerance.

Four configurations, the asymmetric format crossed with both rotations:
    m0_v  ROT_MODE 0, QBITS 3, QJL off   13 words
    m0_k  ROT_MODE 0, QBITS 2, QJL on    14 words
    m1_v  ROT_MODE 1, QBITS 3, QJL off   13 words
    m1_k  ROT_MODE 1, QBITS 2, QJL on    14 words
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")

N = 128


def rd(path, width, signed=True):
    v = np.array([int(l, 16) for l in open(path) if l.strip()], dtype=np.int64)
    if signed:
        v = np.where(v >= (1 << (width - 1)), v - (1 << width), v)
    return v


def codebook(mode, bits):
    raw = rd(os.path.join(RTL, "kv_cb_m%d_b%d.memh" % (mode, bits)), 16)
    nlev = 1 << bits
    return raw[:nlev - 1], raw[nlev - 1:]


def quantize(y, thr_q15, lev_q15, qbits, qjl):
    """Bit-exact model of kv_quantize.sv."""
    sos = int((y.astype(object) ** 2).sum())
    vnorm = math.isqrt(sos)

    thr = [(int(t) * vnorm) >> 15 for t in thr_q15]
    lev = [(int(c) * vnorm) >> 15 for c in lev_q15]

    q = np.array([sum(1 for t in thr if int(yi) >= t) for yi in y], dtype=np.int64)
    rec = np.array([lev[qq] for qq in q], dtype=np.int64)
    resid = y - rec

    rsos = int((resid.astype(object) ** 2).sum())
    rnorm = math.isqrt(rsos)

    nwq = (N * qbits) // 32
    acc = 0
    for qq in q:
        acc = (acc << qbits) | int(qq)
    words = [(acc >> ((nwq - 1 - w) * 32)) & 0xFFFFFFFF for w in range(nwq)]
    words.append(vnorm)

    if qjl:
        sacc = 0
        for r in resid:
            sacc = (sacc << 1) | (1 if r < 0 else 0)
        words += [(sacc >> ((3 - j) * 32)) & 0xFFFFFFFF for j in range(4)]
        words.append(rnorm)

    return words, q, rec, resid, vnorm, rnorm


def append_set():
    """Eight DISTINCT real K vectors for the append/addressing test.

    Distinctness is the whole point: appending one vector eight times
    would pass even if every slot aliased to the same address. These
    are token 0's eight KV heads, so the rows differ genuinely rather
    than by a synthetic perturbation.
    """
    import sys
    sys.path.insert(0, HERE)
    import gen_kv_vectors as G

    _, nkv, _, quant = G._build()
    h_mat, d_vec, _ = G.rot_tables()
    thr_q15, lev_q15 = codebook(0, 2)

    xs, ws = [], []
    for h in range(nkv):
        x = quant["k"][h]                    # token 0, head h
        y = h_mat @ (d_vec * x)
        words, _, _, _, _, _ = quantize(y, thr_q15, lev_q15, 2, 1)
        xs.append(x)
        ws.append(words)

    with open(os.path.join(HERE, "kv_app_x.memh"), "w") as f:
        f.write(chr(10).join("%02x" % (int(v) & 0xFF)
                             for x in xs for v in x) + chr(10))
    with open(os.path.join(HERE, "kv_app_exp.memh"), "w") as f:
        f.write(chr(10).join("%08x" % w for wv in ws for w in wv) + chr(10))

    uniq = len(set(tuple(w) for w in ws))
    print("append set: %d real K vectors (token 0, heads 0..%d), "
          "%d words each, %d distinct packed lines"
          % (len(xs), nkv - 1, len(ws[0]), uniq))
    assert uniq == len(ws), "append vectors are not distinct - test would be vacuous"


def main():
    for mode, rotname in ((0, "had"), (1, "pi")):
        for which, qbits, qjl in (("v", 3, 0), ("k", 2, 1)):
            y = rd(os.path.join(HERE, "kv_%s_%s.memh" % (rotname, which)), 24)
            thr_q15, lev_q15 = codebook(mode, qbits)
            words, q, rec, resid, vnorm, rnorm = quantize(
                y, thr_q15, lev_q15, qbits, qjl)

            name = "kv_q_exp_m%d_%s.memh" % (mode, which)
            with open(os.path.join(HERE, name), "w") as f:
                f.write(chr(10).join("%08x" % w for w in words) + chr(10))

            # Reconstruction error, in the rotated domain. This is a
            # PREVIEW of the step-5 round trip, not a substitute for
            # it: it skips the inverse rotation entirely.
            err = float(np.sqrt((resid.astype(np.float64) ** 2).sum()))
            snr = 20 * math.log10(vnorm / err) if err > 0 else float("inf")
            used = sorted(set(int(x) for x in q))

            print("m%d %s  QBITS %d QJL %d  %2d words  ||y|| %d"
                  % (mode, which, qbits, qjl, len(words), vnorm))
            print("     levels used %s of %d   rotated-domain SNR %.2f dB"
                  % (used, 1 << qbits, snr))
            if qjl:
                print("     residual norm %d  (%.1f%% of ||y||)"
                      % (rnorm, 100.0 * rnorm / vnorm))
    append_set()


if __name__ == "__main__":
    main()
