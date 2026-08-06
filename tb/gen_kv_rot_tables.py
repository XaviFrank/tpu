#!/usr/bin/env python3
"""Generate the two rotation tables for kv_rotate.sv.

Both constructions from docs/kv_cache_design_notes.md section 1 get a
table, because ROT_MODE selects between them and step 5 measures which
one actually wins on this hardware:

    rtl/kv_rot_pi.memh    N*N INT8   dense random orthogonal Pi,
                                     TurboQuant's actual specification
                                     (QR of an i.i.d. Normal matrix)
    rtl/kv_rot_sign.memh  N x 1 bit  Rademacher diagonal D for the
                                     randomized Hadamard path (HD)

THE HADAMARD MATRIX ITSELF IS NOT A TABLE. It is the butterfly wiring
in kv_rotate.sv - 7 stages of add/subtract, no stored coefficients and
no multiplier. Only the random sign flips need storing, 128 bits total
against Pi's 16 KB. That asymmetry IS the cost difference from the
design note, made concrete.

SEED IS FIXED so the RTL tables and the software golden agree. Changing
it invalidates every expected-value file under tb/.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")

N = 128          # head_dim, from software/qwen/config.json
SEED = 0x5B1F    # fixed; see docstring


def haar_orthogonal(d, rng):
    """Uniformly-random (Haar) orthogonal matrix via QR of a Gaussian.

    This is TurboQuant's stated construction, verbatim:
      "generate Pi by applying QR decomposition on a random matrix with
       i.i.d Normal entries"
    The sign correction on R's diagonal is what makes the result
    genuinely Haar-distributed rather than merely orthogonal - numpy's
    QR has a sign convention that biases Q without it.
    """
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def w(path, vals, width):
    mask = (1 << width) - 1
    with open(path, "w") as f:
        f.write(chr(10).join("%0*x" % (width // 4, int(v) & mask)
                             for v in vals) + chr(10))


def main():
    rng = np.random.default_rng(SEED)

    # ---- dense Pi ---------------------------------------------------
    pi = haar_orthogonal(N, rng)
    scale = float(np.abs(pi).max()) / 127.0
    pi_q = np.clip(np.rint(pi / scale), -128, 127).astype(np.int64)

    # ---- Rademacher D for the HD path -------------------------------
    sign = rng.integers(0, 2, size=N)          # 1 means negate

    w(os.path.join(RTL, "kv_rot_pi.memh"), pi_q.reshape(-1), 8)
    w(os.path.join(RTL, "kv_rot_sign.memh"), sign, 8)

    # ---- the number the whole ROT_MODE question turns on ------------
    # Pi is orthogonal in float. Pi_int8 is not, and the residual is a
    # rotation error the Hadamard path does not have at all, because
    # H^T H = N*I holds EXACTLY in integers. Measured here so the
    # step-5 round-trip result can be attributed rather than guessed.
    g = pi_q.astype(np.float64) @ pi_q.astype(np.float64).T
    g /= (np.linalg.norm(pi_q.astype(np.float64), axis=1) ** 2).mean()
    off = g - np.eye(N)
    had = hadamard_check()

    print("dense Pi:")
    print("  entries          min %d max %d  (scale %.6e)"
          % (pi_q.min(), pi_q.max(), scale))
    print("  float orthogonality  max|Pi Pi^T - I|      %.3e"
          % np.abs(pi @ pi.T - np.eye(N)).max())
    print("  INT8  orthogonality  max|off-diagonal|     %.3e" % np.abs(off).max())
    print("  INT8  orthogonality  rms  off-diagonal     %.3e"
          % np.sqrt((off ** 2).sum() / (N * N - N)))
    print("randomized Hadamard:")
    print("  sign flips       %d of %d coordinates negated" % (sign.sum(), N))
    print("  H^T H == N*I     %s  (exact integer identity)" % had)
    print("wrote rtl/kv_rot_pi.memh (%d entries), rtl/kv_rot_sign.memh (%d)"
          % (pi_q.size, N))


def hadamard_check():
    h = hadamard(N)
    return "yes" if np.array_equal(h.T @ h, N * np.eye(N, dtype=np.int64)) else "NO"


def hadamard(n):
    """Sylvester construction, matching the butterfly order in RTL."""
    h = np.ones((1, 1), dtype=np.int64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


if __name__ == "__main__":
    main()
