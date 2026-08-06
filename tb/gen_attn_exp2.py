#!/usr/bin/env python3
"""Generate rtl/exp2_lut.memh - the online-softmax weight table.

ONE OCTAVE ONLY, which is the whole point. Working the softmax in base
2 with an INTEGER running max splits the weight into

    2^(s' - m) = 2^-n * 2^-f     n integer >= 0,  f in [0,1)
                 ^^^^   ^^^^
                 shift  this table

so the table never has to span a range - it covers f in [0,1) and the
integer part is an exact arithmetic shift. That is also why the running
max is kept integral: the rescale factor 2^(m_old - m_new) becomes a
power of two, exact in fixed point, contributing ZERO multiplicative
error to the accumulators it scales.

Q1.15, so 2^-f in (0.5, 1] maps to (16384, 32768]. Q0.16 would put
2^0 = 1.0 at 65536 and overflow the word.

257 ENTRIES, NOT 256. The 257th is the f=1 endpoint that linear
interpolation of the last interval needs. swiglu_gate learned this the
hard way - its table wrapped from the last entry back to entry 0 and
produced a real bug, because SiLU is not periodic. Neither is 2^-f.
(rope's identical-looking wrap is correct, because sine is.)
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")

NENT = 256          # intervals; the table has NENT+1 entries
FRACW = 8           # interpolation fraction bits, matching rope/swiglu


def main():
    tab = [int(round(2.0 ** (-i / float(NENT)) * 32768.0))
           for i in range(NENT + 1)]
    assert tab[0] == 32768 and tab[NENT] == 16384, (tab[0], tab[NENT])

    with open(os.path.join(RTL, "exp2_lut.memh"), "w") as f:
        f.write(chr(10).join("%04x" % v for v in tab) + chr(10))

    # Measure what the hardware will actually do: table lookup plus
    # linear interpolation at FRACW bits, against exact 2^-f.
    worst_abs, worst_rel = 0.0, 0.0
    steps = NENT * (1 << FRACW)
    for t in range(steps):
        i, fr = t >> FRACW, t & ((1 << FRACW) - 1)
        got = tab[i] + (((tab[i + 1] - tab[i]) * fr) >> FRACW)
        exact = 2.0 ** (-(t / float(steps))) * 32768.0
        worst_abs = max(worst_abs, abs(got - exact))
        worst_rel = max(worst_rel, abs(got - exact) / exact)

    print("exp2_lut.memh: %d entries, Q1.15, 2^-f for f in [0,1]" % (NENT + 1))
    print("  endpoints        2^0 -> %d,  2^-1 -> %d" % (tab[0], tab[NENT]))
    print("  interp error     worst abs %.3f LSB   worst rel %.3e (%.1f bits)"
          % (worst_abs, worst_rel, -math.log2(worst_rel)))
    print("  V's 3-bit quantization floor is ~13 dB (2.2 bits), so the")
    print("  table is nowhere near the limiting term.")


if __name__ == "__main__":
    main()
