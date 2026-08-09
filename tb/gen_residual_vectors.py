#!/usr/bin/env python3
"""Real layer-0 residual add: r' = x + o_proj(attn(x)), quantized.

The vectors are the actual residual stream of Qwen3-0.6B layer 0 -
x is the block's input hidden state and o is o_proj's output, exactly
the `xr = x + o` line in attention_l0.forward_layer0 step 9 - captured
with the same transformers hooks calibrate_all.py uses.

FORMATS, and why they differ:
  r  INT16 at s_r = |x|max/32767   the residual stream, wide because it
                                   is accumulated twice per layer
  o  INT8  at s_o = |o|max/127     a stage output at its own requant
                                   scale, which does NOT match s_r
so the hardware must rescale o into r's units before adding, and
(scale_mult, shift_amt) is that ratio as a mantissa and a shift.

The SATURATING case is generated too, by deliberately shrinking the
residual scale until the sum leaves INT16. residual_add's clamp is
reachable on real data with a badly chosen scale - which is exactly
what the flag is for - so the testbench drives it rather than
asserting it cannot happen.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

N = 1024
QMAX = 127
RMAX = 32767
MW, SW = 16, 6


def ms(ratio):
    """ratio as (mantissa, shift), mantissa normalized to MW-1 bits.

    Same form requant/recip/q_scale use. MW-1 because scale_mult is
    SIGNED and the ratio is positive.
    """
    gs = (MW - 2) - int(math.floor(math.log2(ratio)))
    gs = max(0, min(gs, (1 << SW) - 1))
    gm = int(round(ratio * 2 ** gs))
    while gm >= (1 << (MW - 1)) and gs > 0:
        gs -= 1
        gm = int(round(ratio * 2 ** gs))
    return gm, gs


def w(name, vals, width):
    with open(os.path.join(HERE, name), "w") as f:
        f.write("\n".join("%0*x" % (width // 4, int(v) & ((1 << width) - 1))
                          for v in vals) + "\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    text = "In 1969, Apollo 11 landed on the Moon; the crew returned six days later."
    ids = tok(text, return_tensors="pt")["input_ids"]

    grab = {}
    L0 = model.model.layers[0]
    h1 = L0.register_forward_pre_hook(
        lambda m, a: grab.__setitem__("x", a[0].detach().float().numpy()))
    h2 = L0.self_attn.o_proj.register_forward_hook(
        lambda m, a, out: grab.__setitem__("o", out.detach().float().numpy()))
    with torch.no_grad():
        model(ids)
    h1.remove()
    h2.remove()

    # last position of the sequence, the same deterministic rule every
    # other artifact in this project uses
    x = grab["x"].reshape(-1, N)[-1].astype(np.float64)
    o = grab["o"].reshape(-1, N)[-1].astype(np.float64)
    print(f"real layer-0 residual: |x|max {np.abs(x).max():.5f}  "
          f"|o|max {np.abs(o).max():.5f}  |x+o|max {np.abs(x+o).max():.5f}")

    s_r = float(np.abs(x + o).max()) / RMAX     # sized for the SUM, not x
    s_o = float(np.abs(o).max()) / QMAX
    rq = np.clip(np.rint(x / s_r), -RMAX, RMAX).astype(np.int64)
    oq = np.clip(np.rint(o / s_o), -QMAX, QMAX).astype(np.int64)

    gm, gs = ms(s_o / s_r)
    print(f"s_r {s_r:.6e}  s_o {s_o:.6e}  ratio {s_o/s_r:.4f} "
          f"-> mant {gm} shift {gs} (= {gm/2**gs:.6f})")

    def model_add(rv, ov, gm, gs):
        out, sat = [], 0
        for r, ovv in zip(rv, ov):
            p = int(ovv) * gm
            b = (1 << (gs - 1)) if gs > 0 else 0
            s = int(r) + ((p + b) >> gs)
            if s > RMAX:
                s, sat = RMAX, 1
            elif s < -RMAX - 1:
                s, sat = -RMAX - 1, 1
            out.append(s)
        return out, sat

    good, sat0 = model_add(rq, oq, gm, gs)
    assert sat0 == 0, "the well-scaled case should not saturate"
    err = np.abs(np.array(good) * s_r - (x + o)).max()
    print(f"  no saturation; worst |hw - float| = {err:.6e} "
          f"({err/np.abs(x+o).max():.3%} of span)")

    # ---- the saturating case, on purpose --------------------------
    # A residual scale sized for x alone, not for x+o, is a realistic
    # mistake and it overflows: |x+o|max exceeds |x|max.
    s_r2 = float(np.abs(x).max()) / RMAX
    rq2 = np.clip(np.rint(x / s_r2), -RMAX, RMAX).astype(np.int64)
    gm2, gs2 = ms(s_o / s_r2)
    bad, sat1 = model_add(rq2, oq, gm2, gs2)
    nsat = sum(1 for v in bad if v in (RMAX, -RMAX - 1))
    print(f"  under-scaled case: mant {gm2} shift {gs2}, "
          f"saturated {nsat}/{N} elements, flag {sat1}")
    assert sat1 == 1, "the under-scaled case must saturate - the TB needs it"

    w("res_r.memh", rq, 16)
    w("res_o.memh", oq, 8)
    w("res_exp.memh", good, 16)
    w("res_cfg.memh", [gm, gs], 16)
    w("res_r2.memh", rq2, 16)
    w("res_exp2.memh", bad, 16)
    w("res_cfg2.memh", [gm2, gs2], 16)
    print(f"\nwrote 7 .memh files to {HERE}")


if __name__ == "__main__":
    main()
