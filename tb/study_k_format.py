#!/usr/bin/env python3
"""Matched-budget study of the K cache format, scored on ATTENTION
ACCURACY rather than reconstruction SNR.

Why this exists: K's round-trip reconstruction measured 12.03 dB and
looked healthy, but an attention logit is an INNER PRODUCT, and q.k is
a small fraction of |q||k| - so a given relative error on k is
amplified on the logit. The write path chose K's format before any
consumer existed, which made the wrong metric the only measurable one.

UNITS. The scaling here is the one verified in the step-2 diagnostic to
reproduce float logits to ~0.01: yq is pre-multiplied by

    G = sq*sk*log2(e)*2^16 / (N*sqrt(hd))

so that (yq . yhat_K) / 2^16 IS the base-2 logit. The previous attempt
at this comparison invented its own conversion and produced numbers
that contradicted that diagnostic; nothing here is hand-converted.

Three metrics, weakest to strongest:
  logit RMS      base-2 log units, the direct quantity
  weight rel err ||p_hat - p|| / ||p||, softmax over the row
  output rel err ||attn_hat - attn|| / ||attn||, what the layer sees
Only the last one matters for model quality; the first two explain it.

Population: real rows, several sequences x all 8 KV heads, causal, each
token RoPE'd at its own position. One row would not be evidence - that
was the mistake step 2 made with a single crest factor.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)
sys.path.insert(0, HERE)

from fetch_qwen import CACHE, fetch_embed_rows, fetch_file, load_header  # noqa
import attention_l0 as A                                                # noqa
from gen_kv_rot_tables import N, hadamard                               # noqa
from gen_kv_codebooks import lloyd_max                                  # noqa
from gen_kv_quant_vectors import codebook, quantize                     # noqa

NSEQ = 3


def rows():
    """Real causal rows: (q_float, K_float, V_float) per KV head."""
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps, hd, base = cfg["rms_norm_eps"], cfg["head_dim"], cfg["rope_theta"]
    nq, nkv, nf = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["head_dim"] // 2)
    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    th = np.array([base ** (-2.0 * j / hd) for j in range(nf)])
    out = []
    for txt in CALIB_TEXTS[:NSEQ]:
        ids = tok.encode(txt).ids
        T = len(ids)
        rws = fetch_embed_rows(hdr, ds, ids)
        x = np.stack([rws[int(t)] for t in ids]).astype(np.float64)
        h = A.rmsnorm(x, A.T("model.layers.0.input_layernorm.weight"), eps)
        q = A.rmsnorm((h @ A.T("model.layers.0.self_attn.q_proj.weight").T)
                      .reshape(T, nq, hd),
                      A.T("model.layers.0.self_attn.q_norm.weight"), eps)
        k = A.rmsnorm((h @ A.T("model.layers.0.self_attn.k_proj.weight").T)
                      .reshape(T, nkv, hd),
                      A.T("model.layers.0.self_attn.k_norm.weight"), eps)
        v = (h @ A.T("model.layers.0.self_attn.v_proj.weight").T).reshape(T, nkv, hd)
        for t in range(T):
            c, s = np.cos(t * th), np.sin(t * th)
            for arr in (q, k):
                a1, a2 = arr[t, :, :nf].copy(), arr[t, :, nf:].copy()
                arr[t, :, :nf] = a1 * c - a2 * s
                arr[t, :, nf:] = a2 * c + a1 * s
        for kvh in range(nkv):
            out.append((hd, q[T-1, kvh * (nq // nkv)], k[:, kvh], v[:, kvh]))
    return out


def fit_k_book(all_yk, bits):
    """Lloyd-Max on the rotated, normalized K coordinates only."""
    z = []
    for yk in all_yk:
        n = np.linalg.norm(yk)
        if n > 0:
            z.append(yk / n)
    z = np.concatenate(z)
    thr, lev = lloyd_max(z, 1 << bits)
    return (np.clip(np.rint(thr * 32768), -32768, 32767).astype(np.int64),
            np.clip(np.rint(lev * 32768), -32768, 32767).astype(np.int64))


def recon(yk, thr, lev, bits, qjl):
    _, q_, _, _, vn, rn = quantize(yk, thr, lev, bits, qjl)
    rec = np.array([(int(lev[c]) * vn) >> 15 for c in q_], dtype=np.int64)
    if not qjl:
        return rec
    r = yk - rec
    amp = (rn * 2896) >> 15
    return rec + np.where(r < 0, -amp, amp)


def softmax2(l2):
    p = np.exp2(l2 - l2.max())
    return p / p.sum()


def main():
    h_mat = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d_vec = np.where(sign == 1, -1, 1)
    thrV, levV = codebook(0, 3)

    R = rows()
    # rotated K for every row, pooled, to fit the K-only codebooks
    pool = []
    for hd, qf, kf, vf in R:
        sk = float(np.abs(kf).max()) / 127.0
        kq = np.clip(np.rint(kf / sk), -128, 127).astype(np.int64)
        for t in range(kq.shape[0]):
            pool.append((h_mat @ (d_vec * kq[t])).astype(np.float64))

    books = {}
    for b in (2, 3, 4, 5):
        books[b] = fit_k_book(pool, b)
    dep2 = codebook(0, 2)          # the DEPLOYED 2-bit book (K+V pooled)

    cfgs = [("2b MSE  (deployed bk)", 2, 1, dep2, True),
            ("2b MSE  (no QJL)",      2, 0, books[2], False),
            ("2b MSE + 1b QJL",       2, 1, books[2], False),
            ("3b MSE  (no QJL)",      3, 0, books[3], False),
            ("3b MSE + 1b QJL",       3, 1, books[3], False),
            ("4b MSE  (no QJL)",      4, 0, books[4], False),
            ("5b MSE  (no QJL)",      5, 0, books[5], False)]

    print("population: %d real causal rows (%d seqs x 8 KV heads)" % (len(R), NSEQ))
    print("V held at the deployed 3-bit format throughout, so every")
    print("difference below is attributable to K.")
    print("")
    print("%-22s %5s %5s   %-14s %-14s %-14s" %
          ("K format", "words", "b/co", "logit RMS", "weight rel", "OUTPUT rel"))

    floor = None
    for name, bits, qjl, (thr, lev), is_dep in cfgs:
        lg, we, oe = [], [], []
        for hd, qf, kf, vf in R:
            T = kf.shape[0]
            sk = float(np.abs(kf).max()) / 127.0
            sv = float(np.abs(vf).max()) / 127.0
            sq = float(np.abs(qf).max()) / 127.0
            kq = np.clip(np.rint(kf / sk), -128, 127).astype(np.int64)
            vq = np.clip(np.rint(vf / sv), -128, 127).astype(np.int64)
            qq = np.clip(np.rint(qf / sq), -128, 127).astype(np.int64)

            G = sq * sk * math.log2(math.e) * (1 << 16) / (N * math.sqrt(hd))
            yq = np.rint((h_mat @ (d_vec * qq)) * G).astype(np.int64)

            l_true = (qf @ kf.T) * math.log2(math.e) / math.sqrt(hd)
            l_fmt = np.array([float(yq @ recon(h_mat @ (d_vec * kq[t]),
                                               thr, lev, bits, qjl)) / 2**16
                              for t in range(T)])

            p_t, p_f = softmax2(l_true), softmax2(l_fmt)
            # V through the deployed 3-bit path, identically for both
            vh = np.stack([(lambda yv: (h_mat @ recon(yv, thrV, levV, 3, 0))
                            * d_vec / N * sv)(h_mat @ (d_vec * vq[t]))
                           for t in range(T)])
            o_t, o_f = p_t @ (vq * sv), p_f @ vh

            lg.append(np.sqrt(((l_fmt - l_true) ** 2).mean()))
            we.append(np.linalg.norm(p_f - p_t) / np.linalg.norm(p_t))
            oe.append(np.linalg.norm(o_f - o_t) / np.linalg.norm(o_t))

        nwq = (N * bits) // 32
        words = nwq + 1 + (5 if qjl else 0)
        tag = name + (" *" if is_dep else "")
        print("%-22s %5d %5.2f   %.4f         %.4f         %.4f" %
              (tag, words, words * 32.0 / N,
               np.mean(lg), np.mean(we), np.mean(oe)))
        if is_dep:
            floor = np.mean(oe)

    print("")
    print("* = the format currently in the write path (kv_quantize K config).")
    print("Codebooks other than the deployed one are refitted on K ONLY;")
    print("the deployed 2-bit book was fitted on K and V pooled, so the")
    print("gap between the first two rows isolates the refit from the bits.")


if __name__ == "__main__":
    main()
