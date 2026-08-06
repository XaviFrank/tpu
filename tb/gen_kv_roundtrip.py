#!/usr/bin/env python3
"""Round-trip goldens for tb_kv_dequant, and the answer to step 1.

Closes the loop the whole spec exists to close:

    real INT8 x -> rotate -> quantize -> pack -> unpack -> reconstruct
                -> inverse rotate -> x_hat,  compared against x

TWO DIFFERENT COMPARISONS, and conflating them would be the easy
mistake:

  * x_hat against the BIT-EXACT MODEL of kv_dequant.sv - this is a
    correctness check on the hardware, and it must match exactly.
  * x_hat against the ORIGINAL x - this is the accuracy result, and it
    must NOT be exact, because 3-bit quantization is lossy. Reporting a
    small number here is the point; reporting zero would mean the
    testbench was measuring itself.

INVERSE SCALE. The forward path deliberately carries no scale (see
kv_rotate.sv), so the inverse has to put one back:

    Hadamard   H is symmetric with H*H = N*I, so the inverse is the
               SAME butterfly followed by D and a divide by N. N is a
               power of two, so INV_MUL = 1 and the divide is a shift.
               EXACTLY invertible.
    dense Pi   Pi_int8 is only approximately orthogonal - the very
               non-orthogonality measured in step 1 - so the inverse is
               Pi^T followed by a divide by c = mean row norm squared.
               c is not a power of two, hence a constant multiplier.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
sys.path.insert(0, HERE)

from gen_kv_quant_vectors import N, codebook, quantize, rd    # noqa: E402
import gen_kv_vectors as G                                    # noqa: E402

INV_SHIFT = 30


def sat8(v):
    return -128 if v < -128 else (127 if v > 127 else v)


def dequant_model(words, thr_q15, lev_q15, qbits, qjl, mode,
                  h_mat, d_vec, pi_q, inv_mul):
    """Bit-exact model of kv_dequant.sv."""
    nwq = (N * qbits) // 32
    acc = 0
    for w in words[:nwq]:
        acc = (acc << 32) | w
    q = [(acc >> ((N - 1 - i) * qbits)) & ((1 << qbits) - 1) for i in range(N)]

    vnorm = words[nwq]
    lev = [(int(c) * vnorm) >> 15 for c in lev_q15]
    yhat = np.array([lev[qi] for qi in q], dtype=np.int64)

    if qjl:
        sacc = 0
        for w in words[nwq+1:nwq+5]:
            sacc = (sacc << 32) | w
        rnorm = words[nwq + 5]
        # amp = rnorm / sqrt(N), in the same Q1.15 style as the levels
        inv_sqrtn = round(32768.0 / math.sqrt(N))
        amp = (rnorm * inv_sqrtn) >> 15
        for i in range(N):
            s = (sacc >> (N - 1 - i)) & 1
            yhat[i] += -amp if s else amp

    if mode == 0:
        rot = (h_mat @ yhat) * d_vec        # butterfly, then D
    else:
        rot = pi_q.T @ yhat                 # Pi^T

    return np.array([sat8((int(r) * inv_mul) >> INV_SHIFT) for r in rot],
                    dtype=np.int64)


def main():
    _, nkv, _, quant = G._build()
    h_mat, d_vec, pi_q = G.rot_tables()

    c_dense = float((pi_q.astype(np.float64) @ pi_q.astype(np.float64).T)
                    .diagonal().mean())
    inv_mul = {0: (1 << INV_SHIFT) // N,
               1: int(round((1 << INV_SHIFT) / c_dense))}

    print("inverse scale, INV_SHIFT = %d" % INV_SHIFT)
    print("  mode 0  divide by N = %d          INV_MUL %d  (exact, a shift)"
          % (N, inv_mul[0]))
    print("  mode 1  divide by c = %.1f    INV_MUL %d  (Pi_int8 is only"
          " approximately orthogonal)" % (c_dense, inv_mul[1]))
    print("")

    lines = {}
    for mode, rotname in ((0, "had"), (1, "pi")):
        for which, qbits, qjl in (("v", 3, 0), ("k", 2, 1)):
            x = rd(os.path.join(HERE, "kv_x_%s.memh" % which), 8)
            y = rd(os.path.join(HERE, "kv_%s_%s.memh" % (rotname, which)), 24)
            thr_q15, lev_q15 = codebook(mode, qbits)
            words, _, _, _, _, _ = quantize(y, thr_q15, lev_q15, qbits, qjl)
            xh = dequant_model(words, thr_q15, lev_q15, qbits, qjl, mode,
                               h_mat, d_vec, pi_q, inv_mul[mode])

            name = "kv_rt_exp_m%d_%s.memh" % (mode, which)
            with open(os.path.join(HERE, name), "w") as f:
                f.write(chr(10).join("%02x" % (int(v) & 0xFF) for v in xh)
                        + chr(10))

            err = x - xh
            rel = (math.sqrt(float((err.astype(np.float64) ** 2).sum())) /
                   math.sqrt(float((x.astype(np.float64) ** 2).sum())))
            snr = -20 * math.log10(rel) if rel > 0 else float("inf")
            lines[(mode, which)] = snr
            print("m%d %s  round-trip: rel err %.4f  SNR %6.2f dB  "
                  "max |dx| %d  exact %d/%d"
                  % (mode, which, rel, snr, int(np.abs(err).max()),
                     int((err == 0).sum()), N))

    print("")
    print("single vector (head 3), for reference against the testbench:")
    for which in ("k", "v"):
        print("  %s   Hadamard %6.2f dB   dense Pi %6.2f dB   delta %+.2f dB"
              % (which.upper(), lines[(0, which)], lines[(1, which)],
                 lines[(0, which)] - lines[(1, which)]))

    sweep(quant, h_mat, d_vec, pi_q, inv_mul)


def sweep(quant, h_mat, d_vec, pi_q, inv_mul):
    """The step-1 question over the whole population, not one vector.

    Step 2 concluded from a single vector's crest factor and was wrong
    when the same statistic was taken over 816. A round trip on head 3
    is a testbench check, not a result.
    """
    print("")
    print("=== THE STEP-1 QUESTION, SWEPT OVER THE FULL POPULATION ===")
    out = {}
    for mode in (0, 1):
        for which, qbits, qjl in (("v", 3, 0), ("k", 2, 1)):
            thr_q15, lev_q15 = codebook(mode, qbits)
            thr_q15 = np.asarray(thr_q15, dtype=np.int64)
            lev_q15 = np.asarray(lev_q15, dtype=np.int64)
            snrs = []
            for x in quant[which]:
                y = (h_mat @ (d_vec * x)) if mode == 0 else (pi_q @ x)
                vn = math.isqrt(int((y.astype(object) ** 2).sum()))
                thr = (thr_q15 * vn) >> 15
                lev = (lev_q15 * vn) >> 15
                q = np.searchsorted(thr, y, side="right")
                yh = lev[q]
                if qjl:
                    r = y - yh
                    rn = math.isqrt(int((r.astype(object) ** 2).sum()))
                    amp = (rn * round(32768.0 / math.sqrt(N))) >> 15
                    yh = yh + np.where(r < 0, -amp, amp)
                rot = (h_mat @ yh) * d_vec if mode == 0 else pi_q.T @ yh
                xh = np.clip((rot * inv_mul[mode]) >> INV_SHIFT, -128, 127)
                e = float(((x - xh) ** 2).sum())
                s = float((x ** 2).sum())
                if e > 0 and s > 0:
                    snrs.append(10 * math.log10(s / e))
            a = np.array(snrs)
            out[(mode, which)] = a
            print("  mode %d  %s   mean %6.2f dB   sd %4.2f   "
                  "worst %6.2f   best %6.2f   (n=%d)"
                  % (mode, which.upper(), a.mean(), a.std(),
                     a.min(), a.max(), len(a)))

    print("")
    for which in ("k", "v"):
        h, p = out[(0, which)], out[(1, which)]
        print("  %s  Hadamard - dense Pi:  mean %+.2f dB,  worst-case %+.2f dB"
              % (which.upper(), h.mean() - p.mean(), h.min() - p.min()))
    print("")
    print("  spread across K/V (the data-oblivious guarantee):")
    for mode, lbl in ((0, "Hadamard"), (1, "dense Pi")):
        d = abs(out[(mode, "k")].mean() - out[(mode, "v")].mean())
        print("    mode %d %-9s |K-V| = %.2f dB" % (mode, lbl, d))


if __name__ == "__main__":
    main()
