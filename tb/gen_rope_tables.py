#!/usr/bin/env python3
"""Generate RoPE's two tables for rtl/rope.sv.

    rtl/rope_freq.memh   64 x 32-bit per-frequency phase DELTAS
    rtl/rope_sin.memh  4096 x 16-bit sin, Q1.15, one full turn

WHY DELTAS AND NOT A POSITION x FREQUENCY TABLE
  Measured against this model's real config (head_dim 128, base 1e6,
  max_position_embeddings 40960):

      full table, 40960 positions   83.9 Mbit   device has 5.2 Mbit
      full table,  4096 positions    8.4 Mbit   still impossible
      full table,  2048 positions    4.2 Mbit   fits, but consumes 80%
                                                of ALL device BRAM to
                                                cap context at 2048
      incremental (this)            ~70 kbit    full 40960 context

  1205x smaller, and the only option that reaches the model's real
  context length at all. This is not a close tradeoff.

PHASE FORMAT
  Phase is a 32-bit unsigned fraction of ONE TURN, so modulo-2*pi
  wraparound is just integer overflow - no comparison, no subtraction.

  delta_i = round(theta_i / (2*pi) * 2^32),  theta_i = base^(-2i/128)

  Accumulation is integer addition, which is EXACT: phase after n
  positions is exactly n*delta_i mod 2^32, with no drift. The only
  error is the one-time rounding of delta_i itself, which accumulates
  linearly: after 40960 positions the phase is off by at most
  40960 * 0.5 / 2^32 = 4.8e-6 turns = 3.0e-5 rad.

  The tightest delta is the highest-index frequency, delta_63 = 848,
  quantised with 5.9e-4 relative error - that is what sets the bound
  above, and it is why the phase is 32 bits rather than 24.

SIN TABLE
  One table serves BOTH sin and cos: cos(p) = sin(p + quarter turn),
  so cos is the same table read at index + 1024. Free 2x saving.

  4096 entries with linear interpolation gives 2.9e-7 error against
  7.7e-4 without - the same LUT+interp pattern swiglu_gate uses.

  NOTE ON INDEX WRAP: here index+1 wrapping from 4095 back to 0 is
  CORRECT, because sin is periodic. That is the opposite of
  swiglu_gate, where the same wrap was a bug (SiLU is not periodic, so
  it needed a 257th entry). Same code shape, opposite correct answer -
  worth being explicit about.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
CFG = os.path.join(os.path.dirname(HERE), "software", "qwen", "config.json")

PHASE_W = 32
SIN_N = 4096
SIN_F = 15          # Q1.15


def main():
    cfg = json.load(open(CFG))
    hd = cfg["head_dim"]
    base = cfg["rope_theta"]
    maxpos = cfg["max_position_embeddings"]
    nf = hd // 2

    # per-frequency deltas
    deltas = []
    for i in range(nf):
        theta = base ** (-2.0 * i / hd)
        deltas.append(round(theta / (2 * math.pi) * (1 << PHASE_W)))
    with open(os.path.join(RTL, "rope_freq.memh"), "w") as f:
        f.write(chr(10).join("%08x" % d for d in deltas) + chr(10))

    # one turn of sin, Q1.15 (sin = 1.0 saturates by one LSB at 32767)
    sins = []
    for i in range(SIN_N):
        v = math.sin(2 * math.pi * i / SIN_N)
        q = int(round(v * (1 << SIN_F)))
        q = max(-32768, min(32767, q))
        sins.append(q & 0xFFFF)
    with open(os.path.join(RTL, "rope_sin.memh"), "w") as f:
        f.write(chr(10).join("%04x" % s for s in sins) + chr(10))

    # report what the tables actually achieve
    worst_interp = 0.0
    for i in range(SIN_N):
        a = sins[i] - 65536 if sins[i] >= 32768 else sins[i]
        b = sins[(i + 1) % SIN_N]
        b = b - 65536 if b >= 32768 else b
        for f8 in range(0, 256, 16):
            ang = 2 * math.pi * (i + f8 / 256.0) / SIN_N
            approx = (a + ((b - a) * f8) // 256) / (1 << SIN_F)
            worst_interp = max(worst_interp, abs(approx - math.sin(ang)))

    phase_err = maxpos * 0.5 / (1 << PHASE_W)
    print("rope_freq.memh  %d deltas x %d bits" % (nf, PHASE_W))
    print("   theta   %.6f .. %.3e rad/position" % (
        base ** 0.0, base ** (-2.0 * (nf - 1) / hd)))
    print("   delta   %d .. %d" % (deltas[0], deltas[-1]))
    print("   phase error after %d positions: %.2e turns = %.2e rad"
          % (maxpos, phase_err, phase_err * 2 * math.pi))
    print("rope_sin.memh   %d entries x 16 bits (Q1.%d), one turn" % (SIN_N, SIN_F))
    print("   worst interpolation error: %.3e" % worst_interp)
    print("   total table cost: %.1f kbit" % ((nf * PHASE_W * 2 + SIN_N * 16) / 1e3))


if __name__ == "__main__":
    main()
