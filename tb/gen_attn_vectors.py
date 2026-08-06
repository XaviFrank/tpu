#!/usr/bin/env python3
"""Real attention row + bit-exact attn_scan model for tb_attn_scan.

A REAL CAUSAL ROW, not a synthetic one: the first calibration sequence,
each token RoPE'd at its OWN position, query taken at the last token
attending over every earlier position. That matters for section 6 - the
online-softmax rescale only fires when a later position's score beats
the running max, and only a genuine sequence produces that pattern.
The generator counts the rescales and refuses to emit if there are
none, because a scan that never rescales is no evidence the rescale
logic works.

Q head 6 -> KV head 3 (GQA maps j -> j//2), matching attention_l0's
repeat_interleave convention that tb work already verified.

SCALE. kv_rotate sheds all scale, so the query carries it instead:
yQ is pre-multiplied by a gain G chosen so that

    sprime = score >> SSHIFT

lands directly in base-2 logits at Q16.16. G folds the two INT8 cache
scales, the 1/sqrt(head_dim) attention scale, log2(e), and the factor N
that (H*D*q).(H*D*k) = N*(q.k) introduces. Working in base 2 is what
makes the running max integral and the rescale an exact shift.
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
from gen_kv_quant_vectors import codebook, quantize                     # noqa

QHEAD, SSHIFT, SFRAC, GUARD, MAXN = 6, 0, 16, 16, 40


def real_row():
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps, hd, base = cfg["rms_norm_eps"], cfg["head_dim"], cfg["rope_theta"]
    nq, nkv, nf = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["head_dim"] // 2)
    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    ids = tok.encode(CALIB_TEXTS[0]).ids
    T = len(ids)
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    rows = fetch_embed_rows(hdr, ds, ids)

    x = np.stack([rows[int(t)] for t in ids]).astype(np.float64)
    h = A.rmsnorm(x, A.T("model.layers.0.input_layernorm.weight"), eps)
    q = A.rmsnorm((h @ A.T("model.layers.0.self_attn.q_proj.weight").T)
                  .reshape(T, nq, hd),
                  A.T("model.layers.0.self_attn.q_norm.weight"), eps)
    k = A.rmsnorm((h @ A.T("model.layers.0.self_attn.k_proj.weight").T)
                  .reshape(T, nkv, hd),
                  A.T("model.layers.0.self_attn.k_norm.weight"), eps)
    v = (h @ A.T("model.layers.0.self_attn.v_proj.weight").T).reshape(T, nkv, hd)

    # RoPE at each token's OWN position - this is what makes the row real
    th = np.array([base ** (-2.0 * j / hd) for j in range(nf)])
    for t in range(T):
        c, s = np.cos(t * th), np.sin(t * th)
        for arr in (q, k):
            a1, a2 = arr[t, :, :nf].copy(), arr[t, :, nf:].copy()
            arr[t, :, :nf] = a1 * c - a2 * s
            arr[t, :, nf:] = a2 * c + a1 * s

    kvh = QHEAD // (nq // nkv)
    return T, hd, q[T-1, QHEAD], k[:, kvh], v[:, kvh]


def model(yq, entries, tab, levK, levV):
    """Bit-exact model of attn_scan.sv."""
    pk = [[int(yq[i]) * int(levK[kk]) for kk in range(4)] for i in range(N)]
    m, l, o, first, resc = 0, 0, [0]*N, True, 0

    for (cK, sK, vnK, rnK, cV, vnV) in entries:
        dotA = sum(pk[i][cK[i]] for i in range(N))
        dotB = sum(-int(yq[i]) if sK[i] else int(yq[i]) for i in range(N))
        ampk = (rnK * 2896) >> 15
        score = ((dotA * vnK) >> 15) + dotB * ampk
        sp = score >> SSHIFT

        m_cand = (sp >> SFRAC) + 1
        m_next = m_cand if first else max(m_cand, m)
        d = 0 if first else m_next - m
        if d:
            resc += 1
        first, m = False, m_next

        dpos = (m_next << SFRAC) - sp
        nsh, fidx, ffrac = dpos >> SFRAC, (dpos >> 8) & 0xFF, dpos & 0xFF
        ev = tab[fidx] + (((tab[fidx+1] - tab[fidx]) * ffrac) >> 8)
        w = 0 if nsh >= MAXN else (ev << GUARD) >> nsh

        l = (l >> d) + w
        if d:
            o = [x >> d for x in o]
        cw = (w * vnV) >> 15
        for i in range(N):
            o[i] += (cw * int(levV[cV[i]])) >> 15
    return o, l, m, resc


def main():
    T, hd, qv, kv, vv = real_row()
    h_mat = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d_vec = np.where(sign == 1, -1, 1)
    thrK, levK = codebook(0, 2)
    thrV, levV = codebook(0, 3)
    tab = [int(x, 16) for x in open(os.path.join(RTL, "exp2_lut.memh"))]

    sk = float(np.abs(kv).max()) / 127.0
    sv = float(np.abs(vv).max()) / 127.0
    sq = float(np.abs(qv).max()) / 127.0
    kq = np.clip(np.rint(kv / sk), -128, 127).astype(np.int64)
    vq = np.clip(np.rint(vv / sv), -128, 127).astype(np.int64)
    qq = np.clip(np.rint(qv / sq), -128, 127).astype(np.int64)

    # G folds: both cache scales, 1/sqrt(hd), log2(e), and the N from
    # (H D q).(H D k) = N (q.k). Solved for the chosen SSHIFT.
    G = (sq * sk * math.log2(math.e) * (1 << SFRAC) * (1 << SSHIFT)
         / (N * math.sqrt(hd)))
    yq = np.rint((h_mat @ (d_vec * qq)) * G).astype(np.int64)
    assert np.abs(yq).max() < (1 << 23), "yq overflows 24 bits: %d" % np.abs(yq).max()

    entries, kwords, vwords = [], [], []
    for t in range(T):
        yk = h_mat @ (d_vec * kq[t])
        yv = h_mat @ (d_vec * vq[t])
        wk, qk_, _, _, vnK, rnK = quantize(yk, thrK, levK, 2, 1)
        wv, qv_, _, _, vnV, _ = quantize(yv, thrV, levV, 3, 0)
        sK = [1 if r < 0 else 0 for r in (yk - np.array(
            [(int(levK[c]) * vnK) >> 15 for c in qk_]))]
        entries.append((list(map(int, qk_)), sK, vnK, rnK,
                        list(map(int, qv_)), vnV))
        kwords += wk
        vwords += wv

    o, l, m, resc = model(yq, entries, tab, levK, levV)
    assert resc > 0, "no rescale fired - the test would be vacuous"

    def w(name, vals, width):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(chr(10).join("%0*x" % (width // 4, int(x) & ((1 << width) - 1))
                                 for x in vals) + chr(10))

    w("attn_yq.memh", yq, 24)
    w("attn_kw.memh", kwords, 32)
    w("attn_vw.memh", vwords, 32)
    w("attn_o.memh", o, 48)
    w("attn_l.memh", [l], 48)

    # float reference: real attention for this head over this row
    ref = A.rmsnorm  # keep the import meaningful
    sc = (qq @ kq.T) * sq * sk / math.sqrt(hd)
    p = np.exp(sc - sc.max()); p /= p.sum()
    ref_out = p @ (vq * sv)
    got = (np.array(o, dtype=np.float64) / l) * (1 << 15)
    got = (h_mat @ got) * d_vec / N * sv          # un-rotate, back to real units
    rel = (np.linalg.norm(got - ref_out) / np.linalg.norm(ref_out))

    print("real row: %d positions, Q head %d -> KV head %d" % (T, QHEAD, QHEAD // 2))
    print("  rescale events   %d of %d positions  (max moved %d times)" % (resc, T, resc))
    print("  final m %d   l %d" % (m, l))
    print("  vs float attention: rel err %.4f  (%.2f dB)"
          % (rel, -20 * math.log10(rel)))
    print("  wrote attn_yq/kw/vw/o/l .memh")


if __name__ == "__main__":
    main()
