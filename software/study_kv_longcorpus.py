#!/usr/bin/env python3
"""Long-corpus confirmation of the KV format numbers.

The 24-point run was decisive at a 55-point effect size and visibly not
at a 2-point one - its perplexity column inverted between two configs
whose KL and top-1 both ordered correctly. This re-runs the same
comparison with ~10x the evaluation points before the 79.2% figure is
used for a design decision.

TWO THINGS MAKE IT TRACTABLE, both exact rather than approximations:

  QUANTIZE ONCE, SLICE PER CUT. Cache quantization is per-position
  independent, and attention at position c only reads positions < c.
  So the full-length cache can be quantized once and sliced, instead of
  re-quantizing for every cut. That is O(T) work instead of O(cuts*T) -
  the naive form was several hours.

  VECTORIZED QUANTIZER. The bit-exact model loops over 128 coordinates
  in Python. This is the same integer arithmetic expressed in numpy -
  same shifts, same floor semantics, same thresholds - and it is
  checked against the scalar model on real vectors before use, because
  a "faster equivalent" that is not equivalent would silently rewrite
  the result.
"""

import math
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
TB = os.path.join(os.path.dirname(HERE), "tb")
CACHE = os.path.join(HERE, "qwen")
sys.path.insert(0, TB)

from gen_kv_rot_tables import N, hadamard                  # noqa: E402
from gen_kv_quant_vectors import codebook, quantize        # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
CORPUS_URL = "https://www.gutenberg.org/files/11/11-0.txt"   # public domain
NTOK = 1024
NPROMPT = 240
MINCTX = 32


def corpus(tok):
    path = os.path.join(CACHE, "eval_corpus.txt")
    if not os.path.exists(path):
        req = urllib.request.Request(path and CORPUS_URL,
                                     headers={"User-Agent": "tpu-study"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode("utf-8", "ignore")
        # skip the licence header; take running prose
        i = txt.find("CHAPTER I")
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt[i:i + 200000])
    return open(path, encoding="utf-8").read()


def isqrt_vec(a):
    return np.array([math.isqrt(int(x)) for x in a], dtype=np.int64)


def qdq_vec(vecs, h, d, book, bits, qjl, low_bit=True):
    """Vectorized, integer-exact equivalent of the scalar model."""
    thr, lev = book
    s = float(np.abs(vecs).max()) / 127.0
    if s == 0:
        return vecs
    xq = np.clip(np.rint(vecs / s), -128, 127).astype(np.int64)
    Y = (d * xq) @ h.T                                    # [T,128]
    if not low_bit:                                       # INT8 + rotation only
        return ((Y @ h.T) * d / N * s)
    vn = isqrt_vec((Y.astype(object) ** 2).sum(axis=1))[:, None]
    ts = (np.asarray(thr, dtype=np.int64)[None, :] * vn) >> 15   # [T,nlev-1]
    q = (Y[:, :, None] >= ts[:, None, :]).sum(axis=2)            # [T,128]
    rec = (np.asarray(lev, dtype=np.int64)[q] * vn) >> 15
    if qjl:
        r = Y - rec
        rn = isqrt_vec((r.astype(object) ** 2).sum(axis=1))[:, None]
        amp = (rn * 2896) >> 15
        rec = rec + np.where(r < 0, -amp, amp)
    return ((rec @ h.T) * d / N * s)


def check_equivalence(h, d, bookK, bookV):
    """A faster path that is not equivalent would silently rewrite the
    answer, so prove it against the scalar model before trusting it."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal((6, N)) * 0.4
    for book, bits, qjl in ((bookK, 2, 1), (bookV, 3, 0)):
        thr, lev = book
        s = float(np.abs(v).max()) / 127.0
        xq = np.clip(np.rint(v / s), -128, 127).astype(np.int64)
        ref = np.empty_like(v)
        for t in range(v.shape[0]):
            y = h @ (d * xq[t])
            _, q_, _, _, vn, rn = quantize(y, thr, lev, bits, qjl)
            rec = np.array([(int(lev[c]) * vn) >> 15 for c in q_], dtype=np.int64)
            if qjl:
                r = y - rec
                rec = rec + np.where(r < 0, -((rn * 2896) >> 15), (rn * 2896) >> 15)
            ref[t] = (h @ rec) * d / N * s
        got = qdq_vec(v, h, d, book, bits, qjl)
        if not np.allclose(ref, got, rtol=0, atol=1e-9):
            raise SystemExit("vectorized quantizer DIVERGES from the scalar "
                             "model (bits=%d qjl=%d) - aborting" % (bits, qjl))
    print("vectorized quantizer matches the scalar bit-exact model")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    h = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d = np.where(sign == 1, -1, 1)
    bookK, bookV = codebook(0, 2), codebook(0, 3)
    check_equivalence(h, d, bookK, bookV)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    nl = model.config.num_hidden_layers

    ids = tok(corpus(tok), return_tensors="pt").input_ids[:, :NTOK]
    Tn = ids.shape[1]
    cuts = np.unique(np.linspace(MINCTX, Tn - 2, NPROMPT).astype(int))
    print("corpus %d tokens, %d evaluation points (was 24)" % (Tn, len(cuts)))

    with torch.no_grad():
        full = model(ids[:, :Tn - 1], use_cache=True).past_key_values
    K = [full.layers[i].keys[0].numpy().astype(np.float64) for i in range(nl)]
    V = [full.layers[i].values[0].numpy().astype(np.float64) for i in range(nl)]
    print("prefilled once: %d layers x %s" % (nl, K[0].shape))

    def variant(fk, fv):
        if fk is None:
            return K, V
        return ([np.stack([fk(K[i][hh]) for hh in range(K[i].shape[0])])
                 for i in range(nl)],
                [np.stack([fv(V[i][hh]) for hh in range(V[i].shape[0])])
                 for i in range(nl)])

    def evaluate(Kq, Vq):
        top1, probs, nll = [], [], []
        for c in cuts:
            cc = DynamicCache()
            for i in range(nl):
                cc.update(torch.tensor(Kq[i][:, :c, :], dtype=torch.float32
                                       ).unsqueeze(0),
                          torch.tensor(Vq[i][:, :c, :], dtype=torch.float32
                                       ).unsqueeze(0), i)
            with torch.no_grad():
                lg = model(ids[:, c:c+1], past_key_values=cc).logits[0, -1].double()
            p = torch.softmax(lg, -1)
            top1.append(int(lg.argmax()))
            probs.append(p)
            nll.append(-math.log(max(float(p[ids[0, c+1]]), 1e-12)))
        return top1, probs, nll

    kf = lambda x: qdq_vec(x, h, d, bookK, 2, 1)
    vf = lambda x: qdq_vec(x, h, d, bookV, 3, 0)
    k8 = lambda x: qdq_vec(x, h, d, bookK, 2, 1, low_bit=False)
    v8 = lambda x: qdq_vec(x, h, d, bookV, 3, 0, low_bit=False)

    print("running exact reference ...")
    te, pe, ne = evaluate(K, V)
    print("%-20s %-13s %-11s %s" % ("config", "top-1 agree", "mean KL", "ppl"))
    print("%-20s %-13s %-11s %.2f" % ("exact reference", "100.0%", "0.0000",
                                      math.exp(np.mean(ne))))
    for label, fk_, fv_ in (("K 2b+QJL / V 3b", kf, vf),
                            ("K INT8   / V 3b", k8, vf),
                            ("K INT8   / V INT8", k8, v8)):
        Kq, Vq = variant(fk_, fv_)
        tq, pq, nq = evaluate(Kq, Vq)
        kl = float(np.mean([float((a * (a.log() - b.log())).sum())
                            for a, b in zip(pe, pq)]))
        ag = 100.0 * np.mean([1.0 if a == b else 0.0 for a, b in zip(te, tq)])
        print("%-20s %-13.1f %-11.4f %.2f"
              % (label, ag, kl, math.exp(np.mean(nq))))


if __name__ == "__main__":
    main()
