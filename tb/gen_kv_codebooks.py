#!/usr/bin/env python3
"""Fit the scalar quantizer codebooks for kv_quantize.sv.

ONE CODEBOOK PER ROT_MODE, and that is not tidiness - it is required
for the step-5 comparison to mean anything. A Haar rotation makes
coordinates ~Gaussian (the Beta density TurboQuant's Lloyd-Max
quantizer is fitted to). The Hadamard path spreads energy
deterministically and comes out FLATTER than Gaussian. Scoring the
Hadamard path with a Beta-fitted codebook would charge it for a
distribution mismatch that has nothing to do with the rotation. See
docs/kv_cache_design_notes.md section 3.

Two bit-widths per mode, because the format is asymmetric:
    V  -> pure 3-bit MSE          (V is consumed as a weighted sum,
                                   reconstruction error is what counts)
    K  -> 2-bit MSE + 1-bit QJL   (K is consumed as an inner product,
                                   and MSE-optimal quantizers are
                                   BIASED for inner products - the
                                   paper's own reason for the QJL stage)

Everything is emitted in Q1.15 so the hardware compare path is exact
integer arithmetic: threshold_k = (t_k * ||y||) >> 15, compared against
the 24-bit rotated coordinate. No division and no rounding convention
anywhere in the quantize datapath.

Emits into rtl/:
    kv_cb_m<mode>_b<bits>.memh    (2^b - 1) thresholds then 2^b levels
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

from gen_kv_rot_tables import N, hadamard                    # noqa: E402
import gen_kv_vectors as G                                   # noqa: E402


def lloyd_max(samples, nlev, iters=200):
    """1-D Lloyd-Max. Initialised on quantiles so it starts inside the
    data rather than on a grid the tails would drag around."""
    s = np.sort(samples)
    c = np.quantile(s, (np.arange(nlev) + 0.5) / nlev)
    for _ in range(iters):
        t = (c[:-1] + c[1:]) / 2.0
        idx = np.searchsorted(t, s)
        new = c.copy()
        for k in range(nlev):
            m = idx == k
            if m.any():
                new[k] = s[m].mean()
        if np.allclose(new, c):
            break
        c = new
    return (c[:-1] + c[1:]) / 2.0, c


def q15(v):
    """Q1.15, saturating. +-1.0 is representable as +-32767."""
    return int(np.clip(np.rint(np.asarray(v) * 32768.0), -32768, 32767))


def main():
    cfg, vecs = G.load_real_kv()
    h_mat = hadamard(N)
    sign = np.array([int(l, 16) for l in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d_vec = np.where(sign == 1, -1, 1)
    pi_q = np.array([int(l, 16) for l in
                     open(os.path.join(RTL, "kv_rot_pi.memh"))], dtype=np.int64)
    pi_q = np.where(pi_q > 127, pi_q - 256, pi_q).reshape(N, N)

    print("fitting on %d real K vectors + %d real V vectors"
          % (len(vecs["k"]), len(vecs["v"])))

    for mode, label in ((0, "randomized Hadamard"), (1, "dense Pi")):
        pooled, per_tensor = [], {}
        for name in ("k", "v"):
            q = vecs[name]
            y = (q * d_vec) @ h_mat.T if mode == 0 else q @ pi_q.T
            nrm = np.sqrt((y.astype(np.float64) ** 2).sum(axis=1, keepdims=True))
            z = y / np.where(nrm > 0, nrm, 1.0)
            per_tensor[name] = z.reshape(-1)
            pooled.append(z.reshape(-1))
        pooled = np.concatenate(pooled)

        print("\n=== ROT_MODE %d (%s) ===" % (mode, label))
        print("  normalized coords: sd %.5f  peak/rms %.3f  (Gaussian ref 1/sqrt(N)=%.5f)"
              % (pooled.std(), np.abs(pooled).max() / pooled.std(),
                 1.0 / np.sqrt(N)))

        for bits in (2, 3):
            thr, lev = lloyd_max(pooled, 1 << bits)
            rec = lev[np.searchsorted(thr, pooled)]
            snr = 10 * np.log10((pooled ** 2).sum() / ((pooled - rec) ** 2).sum())

            # Does one pooled codebook serve both tensors, or is K/V
            # divergence large enough to need separate books?
            spread = []
            for name in ("k", "v"):
                zz = per_tensor[name]
                rr = lev[np.searchsorted(thr, zz)]
                spread.append(10 * np.log10((zz ** 2).sum() / ((zz - rr) ** 2).sum()))

            path = os.path.join(RTL, "kv_cb_m%d_b%d.memh" % (mode, bits))
            with open(path, "w") as f:
                f.write(chr(10).join(
                    ["%04x" % (q15(t) & 0xFFFF) for t in thr] +
                    ["%04x" % (q15(c) & 0xFFFF) for c in lev]) + chr(10))

            print("  %d-bit  SNR %5.2f dB   (K %5.2f, V %5.2f)   -> %s"
                  % (bits, snr, spread[0], spread[1], os.path.basename(path)))
            print("         levels " + " ".join("%+.4f" % c for c in lev))


if __name__ == "__main__":
    main()
