#!/usr/bin/env python3
"""Generate rtl/rsqrt_lut.memh - the initial 1/sqrt estimate for rmsnorm.

Same pattern as gen_requant_vectors.py: an independent Python model
produces the constants, the RTL consumes them, and the TB checks the
hardware against real math rather than against a re-derivation.

FORMAT
  The rsqrt stage normalises its 48-bit input M to
      norm = M >> sh,  sh even,  norm in [2^30, 2^32)
  so u = norm / 2^30 lies in [1, 4). The LUT is indexed by the top 8
  bits of norm (norm[31:24] = floor(u * 64)), giving 192 live entries
  out of 256 (indices 0..63 are unreachable because norm >= 2^30).

  Entry i is  round(2^17 / sqrt(u_mid))  with u_mid = (i + 0.5) / 64,
  the bin midpoint, which halves the worst-case initial error versus
  using the bin's left edge.

  Output range is [2^16, 2^17] -> 18 bits unsigned, Q1.17.

ACCURACY
  8 index bits give the seed about 7-8 correct bits; one
  Newton-Raphson step roughly doubles that, so the refined result lands
  near 15-16 bits. tb_rmsnorm measures the ACTUAL error against real
  1/sqrt over a swept range rather than trusting this estimate.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "rtl", "rsqrt_lut.memh")

ENTRIES = 256
FRAC = 17          # Q1.17


def main():
    lines = []
    for i in range(ENTRIES):
        if i < 64:
            # unreachable: norm >= 2^30 means index >= 64. Fill with the
            # u=1 value so a wild index still yields something sane
            # rather than X.
            y = (1 << FRAC)
        else:
            u_mid = (i + 0.5) / 64.0
            y = int(round((1 << FRAC) / math.sqrt(u_mid)))
        y = max(0, min((1 << 18) - 1, y))
        lines.append(f"{y:05x}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # report the seed's own accuracy, before Newton-Raphson
    worst = 0.0
    for i in range(64, ENTRIES):
        u_mid = (i + 0.5) / 64.0
        y = int(round((1 << FRAC) / math.sqrt(u_mid)))
        for frac in (0.0, 0.5, 0.999):
            u = (i + frac) / 64.0
            exact = 1.0 / math.sqrt(u)
            worst = max(worst, abs(y / (1 << FRAC) - exact) / exact)
    print(f"wrote {OUT} ({ENTRIES} entries, Q1.{FRAC})")
    print(f"seed worst-case relative error before NR: {worst:.5f} "
          f"(~{-math.log2(worst):.1f} correct bits)")


if __name__ == "__main__":
    main()
