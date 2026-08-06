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

QHEADS = (6, 7)          # both Q heads that share KV head 3
SSHIFT, SFRAC, GUARD, MAXN = 0, 16, 16, 40


def real_row():
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps, hd, base = cfg["rms_norm_eps"], cfg["head_dim"], cfg["rope_theta"]
    nq, nkv, nf = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["head_dim"] // 2)
    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    # All the calibration text, not one short sentence: with K now
    # exact the logits no longer carry quantization noise, and a
    # 10-position row never crossed an integer max boundary. The
    # old test's single rescale was partly noise-driven.
    ids = tok.encode(" ".join(CALIB_TEXTS)).ids
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

    kvh = QHEADS[0] // (nq // nkv)
    assert all(qh // (nq // nkv) == kvh for qh in QHEADS)
    return T, hd, [q[T-1, qh] for qh in QHEADS], k[:, kvh], v[:, kvh]


def model(yq, entries, tab, levV):
    """Bit-exact model of attn_scan.sv, K at INT8."""
    m, l, o, first, resc = 0, 0, [0]*N, True, 0

    for (kv8, cV, vnV) in entries:
        # K is plain INT8 and unrotated now: no codebook, no per-vector
        # norm, no QJL residual. One real MAC per coordinate.
        score = sum(int(yq[i]) * int(kv8[i]) for i in range(N))
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
    T, hd, qvs, kv, vv = real_row()
    h_mat = hadamard(N)
    sign = np.array([int(x, 16) for x in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d_vec = np.where(sign == 1, -1, 1)
    thrV, levV = codebook(0, 3)
    tab = [int(x, 16) for x in open(os.path.join(RTL, "exp2_lut.memh"))]

    sk = float(np.abs(kv).max()) / 127.0
    sv = float(np.abs(vv).max()) / 127.0
    kq = np.clip(np.rint(kv / sk), -128, 127).astype(np.int64)
    vq = np.clip(np.rint(vv / sv), -128, 127).astype(np.int64)

    # Cache lines are built ONCE and reused by every sharing Q head -
    # which is the whole point of the GQA reuse, made literal here.
    entries, kwords, vwords = [], [], []
    for t in range(T):
        acc = 0
        for val in kq[t]:
            acc = (acc << 8) | (int(val) & 0xFF)
        kwords += [(acc >> ((31 - w) * 32)) & 0xFFFFFFFF for w in range(32)]
        yv = h_mat @ (d_vec * vq[t])
        wv, qv_, _, _, vnV, _ = quantize(yv, thrV, levV, 3, 0)
        entries.append((list(map(int, kq[t])), list(map(int, qv_)), vnV))
        vwords += wv

    def w(name, vals, width):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(chr(10).join("%0*x" % (width // 4, int(x) & ((1 << width) - 1))
                                 for x in vals) + chr(10))

    w("attn_kw.memh", kwords, 32)
    w("attn_vw.memh", vwords, 32)
    # Raw INT8 vectors too, so the integration TB can drive the REAL
    # write path (kv_pack_int8 for K, kv_rotate+kv_quantize for V)
    # rather than injecting pre-packed words.
    w("attn_xk.memh", kq.reshape(-1), 8)
    w("attn_xv.memh", vq.reshape(-1), 8)
    print("real row: %d positions, KV head %d shared by Q heads %s"
          % (T, QHEADS[0] // 2, list(QHEADS)))

    for hi, qv in enumerate(qvs):
        sq = float(np.abs(qv).max()) / 127.0
        qq = np.clip(np.rint(qv / sq), -128, 127).astype(np.int64)
        G = (sq * sk * math.log2(math.e) * (1 << SFRAC) * (1 << SSHIFT)
             / math.sqrt(hd))
        yq = np.rint(qq * G).astype(np.int64)
        assert np.abs(yq).max() < (1 << 31), "yq overflows 32 bits"

        # Each head is modelled INDEPENDENTLY - no shared state - so the
        # testbench comparing the shared hardware against these goldens
        # is exactly section 6's "matches independent computation".
        o, l, m, resc = model(yq, entries, tab, levV)
        assert resc > 0, "no rescale fired for head %d" % QHEADS[hi]

        w("attn_yq%d.memh" % hi, yq, 32)
        w("attn_o%d.memh" % hi, o, 48)
        w("attn_l%d.memh" % hi, [l], 48)

        sc = (qq @ kq.T) * sq * sk / math.sqrt(hd)
        pr = np.exp(sc - sc.max()); pr /= pr.sum()
        ref_out = pr @ (vq * sv)
        got = (np.array(o, dtype=np.float64) / l) * (1 << 15)
        got = (h_mat @ got) * d_vec / N * sv
        rel = np.linalg.norm(got - ref_out) / np.linalg.norm(ref_out)
        print("  Q head %d: %d rescale(s), m %d, vs float %.2f dB"
              % (QHEADS[hi], resc, m, -20 * math.log10(rel)))


if __name__ == "__main__":
    main()
