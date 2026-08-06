#!/usr/bin/env python3
"""Does the KV cache format actually change what the model emits?

Every earlier study measured attention-internal quantities on layer 0:
logit error, weight error, attention-output error. All of them said the
K format is badly degraded (output relative error ~0.92 at best against
a 0.14 V-only control). None of them can say whether the MODEL cares -
early layers often attend broadly, and a wrong near-uniform row may
cost little end to end.

This runs the real Qwen3-0.6B and compares the next-token distribution
with an exact KV cache against one that has been round-tripped through
our format. The unhooked model IS the reference, so ground truth is
free by construction - which is exactly why this is worth the torch
dependency that fetch_qwen.py was written to avoid. Answering "does
model quality degrade" from a hand-rolled 28-layer reimplementation
with nothing to check it against would be the unfalsifiable-claim trap
this project keeps catching.

LAYER 0 ONLY vs ALL LAYERS separates two questions in one run:
  layer-0-only degradation ~ 0   -> the measured attention error is
                                    real but does not matter, and the
                                    format ships
  all-layers degradation   large -> the format must change, and by how
                                    much is now measurable

The quantizer is the SAME Python model already verified bit-exact
against the RTL (gen_kv_quant_vectors.quantize), so this measures the
shipped format, not an idealization of it.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
TB = os.path.join(os.path.dirname(HERE), "tb")
sys.path.insert(0, TB)

from gen_kv_rot_tables import N, hadamard                  # noqa: E402
from gen_kv_quant_vectors import codebook, quantize        # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
NPROMPT = 24          # evaluation points
MINCTX = 16


def load_tables():
    h = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d = np.where(sign == 1, -1, 1)
    return h, d, codebook(0, 2), codebook(0, 3)


def qdq_identity(vecs, h, d, book, bits, qjl):
    """Plumbing control: same numpy/torch round trip, no quantization.
    Must give 100% top-1 agreement, or the cache surgery is the bug and
    every number below is measuring my hook rather than the format."""
    return vecs


def qdq_int8(vecs, h, d, book, bits, qjl):
    """Isolates INT8 + rotate + inverse-rotate, with NO codebook stage.
    Brackets the result: whatever this costs is not the low-bit format."""
    s = float(np.abs(vecs).max()) / 127.0
    if s == 0:
        return vecs
    xq = np.clip(np.rint(vecs / s), -128, 127).astype(np.int64)
    out = np.empty_like(vecs)
    for t in range(vecs.shape[0]):
        out[t] = (h @ (h @ (d * xq[t]))) * d / N * s
    return out


def qdq(vecs, h, d, book, bits, qjl):
    """Full round trip through the shipped format, one head's cache.

    vecs: [T, 128] float. Returns the same shape after INT8 ->
    rotate -> low-bit quantize -> reconstruct -> inverse rotate.
    """
    thr, lev = book
    T = vecs.shape[0]
    s = float(np.abs(vecs).max()) / 127.0
    if s == 0:
        return vecs
    xq = np.clip(np.rint(vecs / s), -128, 127).astype(np.int64)
    out = np.empty_like(vecs)
    for t in range(T):
        y = h @ (d * xq[t])
        _, q_, _, _, vn, rn = quantize(y, thr, lev, bits, qjl)
        rec = np.array([(int(lev[c]) * vn) >> 15 for c in q_], dtype=np.int64)
        if qjl:
            r = y - rec
            amp = (rn * 2896) >> 15
            rec = rec + np.where(r < 0, -amp, amp)
        out[t] = (h @ rec) * d / N * s
    return out


# The Cache object moved from parallel key_cache/value_cache lists to a
# per-layer object across transformers versions. Probe rather than pin.
def get_kv(cache, li, which):
    if hasattr(cache, "layers"):
        lay = cache.layers[li]
        return lay.keys if which == "k" else lay.values
    return (cache.key_cache if which == "k" else cache.value_cache)[li]


def set_kv(cache, li, which, val):
    if hasattr(cache, "layers"):
        lay = cache.layers[li]
        if which == "k":
            lay.keys = val
        else:
            lay.values = val
    else:
        (cache.key_cache if which == "k" else cache.value_cache)[li] = val


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    h, d, bookK, bookV = load_tables()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    nl = model.config.num_hidden_layers
    print("loaded %s: %d layers, head_dim %d, %d KV heads"
          % (MODEL, nl, model.config.head_dim, model.config.num_key_value_heads))

    from quantize_pack import CALIB_TEXTS
    text = " ".join(CALIB_TEXTS)
    ids = tok(text, return_tensors="pt").input_ids
    Tn = ids.shape[1]
    cuts = np.linspace(MINCTX, Tn - 2, NPROMPT).astype(int)
    print("eval text %d tokens, %d cut positions from %d to %d"
          % (Tn, len(cuts), cuts[0], cuts[-1]))
    print("")

    def run(layers, fn=None):
        """layers=None -> exact. Otherwise quantize those layers' cache."""
        kls, top1, nll = [], [], []
        for cut in cuts:
            with torch.no_grad():
                pre = model(ids[:, :cut], use_cache=True)
                cache = pre.past_key_values
                if layers is not None:
                    for li in layers:
                        k = get_kv(cache, li, "k")[0].numpy().astype(np.float64)
                        v = get_kv(cache, li, "v")[0].numpy().astype(np.float64)
                        for hd in range(k.shape[0]):
                            f = fn or qdq
                            k[hd] = f(k[hd], h, d, bookK, 2, 1)
                            v[hd] = f(v[hd], h, d, bookV, 3, 0)
                        set_kv(cache, li, "k", torch.tensor(
                            k, dtype=torch.float32).unsqueeze(0))
                        set_kv(cache, li, "v", torch.tensor(
                            v, dtype=torch.float32).unsqueeze(0))
                out = model(ids[:, cut:cut+1], past_key_values=cache,
                            use_cache=True)
                lg = out.logits[0, -1].double()
            yield_p = torch.softmax(lg, -1)
            nll.append(-math.log(max(float(yield_p[ids[0, cut+1]]), 1e-12)))
            top1.append(int(lg.argmax()))
            kls.append(yield_p)
        return kls, top1, nll

    print("running exact reference ...")
    pe, te, ne = run(None)

    trials = [("CONTROL identity", list(range(nl)), qdq_identity),
              ("CONTROL int8+rot", list(range(nl)), qdq_int8),
              ("layer 0 only", [0], None),
              ("ALL %d layers" % nl, list(range(nl)), None)]
    for label, layers, fn in trials:
        print("running %s ..." % label)
        pq, tq, nq = run(layers, fn)
        kl = [float((a * (a.log() - b.log())).sum()) for a, b in zip(pe, pq)]
        agree = np.mean([1.0 if x == y else 0.0 for x, y in zip(te, tq)])
        print("  %-16s top-1 agreement %.1f%%   mean KL %.4f nats   "
              "NLL %.4f -> %.4f  (ppl %.2f -> %.2f)"
              % (label, 100 * agree, float(np.mean(kl)),
                 np.mean(ne), np.mean(nq),
                 math.exp(np.mean(ne)), math.exp(np.mean(nq))))


if __name__ == "__main__":
    main()
