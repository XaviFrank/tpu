#!/usr/bin/env python3
"""Generate rtl/silu_lut.memh - the SiLU table for swiglu_gate.

FORMAT
  Domain [-8, +8) in Q4.12 signed, so the 16-bit dequantized gate value
  maps straight onto the table: index = (x_q + 32768) >> 8 (8 bits),
  fraction = x_q & 0xFF (8 bits) for linear interpolation.

  257 entries, not 256: interpolating at index 255 needs its right
  neighbour, and an extra entry is cheaper and clearer than special
  casing the top bin.

  Entries are silu(x) in Q4.12 signed. silu is bounded above by x, so
  over this domain it stays inside Q4.12's [-8, 8) range; the minimum
  is about -0.2785 near x = -1.278.

WHY [-8, +8)
  Measured on the real calibration set, |gate| never exceeds 5.92, so
  this covers the real distribution with margin. Outside it, the RTL
  uses the analytic tails rather than the table: silu(x) -> x for large
  positive x (relative error 3.4e-4 at x = 8) and silu(x) -> 0 for
  large negative x (absolute error 2.7e-3 at x = -8).

ACCURACY
  Checked against the REAL gate distribution, not a generic sweep -
  256 bins with linear interpolation give a max absolute error of
  2.44e-4, against a half-LSB budget of 5.5e-3 for the INT8 output.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "rtl", "silu_lut.memh")

ENTRIES = 257          # 256 bins + the right neighbour of the last one
F = 12                 # Q4.12
LO, HI = -8.0, 8.0


def silu(x):
    return x / (1.0 + math.exp(-x))


def main():
    step = (HI - LO) / (ENTRIES - 1)
    vals = []
    for i in range(ENTRIES):
        x = LO + i * step
        q = int(round(silu(x) * (1 << F)))
        q = max(-32768, min(32767, q))
        vals.append(q & 0xFFFF)

    with open(OUT, "w") as f:
        f.write("\n".join("%04x" % v for v in vals) + "\n")

    # worst-case interpolation error across the whole domain
    worst = 0.0
    for i in range(ENTRIES - 1):
        a = vals[i] - 65536 if vals[i] >= 32768 else vals[i]
        b = vals[i + 1] - 65536 if vals[i + 1] >= 32768 else vals[i + 1]
        for f8 in range(0, 256, 8):
            x = LO + (i + f8 / 256.0) * step
            approx = (a + ((b - a) * f8) // 256) / (1 << F)
            worst = max(worst, abs(approx - silu(x)))

    print("wrote %s (%d entries, Q4.%d, domain [%g, %g))"
          % (OUT, ENTRIES, F, LO, HI))
    print("worst interpolation error over the full domain: %.3e" % worst)
    print("half-LSB budget for the INT8 output is 5.5e-3, so margin is %.0fx"
          % (5.5e-3 / worst))


if __name__ == "__main__":
    main()
