#!/usr/bin/env python3
"""Generate real K/V vectors and rotation goldens for tb_kv_rotate.

Vectors are taken exactly where the KV cache write path sits: K is
post-k_norm AND post-RoPE, V is raw v_proj output. That asymmetry is
the model's, not a simplification - Qwen3 norms Q and K over head_dim
but NOT V, and RoPE applies to Q and K only. So K and V arrive at the
cache with genuinely different distributions, which is exactly what
section 4's non-vacuity requirement wants exercised.

BOTH ROTATIONS ARE EXACT INTEGER MAPS, so the testbench can demand
bit-exact agreement rather than a tolerance:

    Hadamard   y = H_unnorm @ (D * x)        entries +-1, sums only
    dense      y = Pi_int8 @ x               INT8 x INT8 -> 24-bit acc

Neither golden needs a rounding convention, and no scale is involved:
the 1/sqrt(N) that makes H orthonormal, and Pi's own quantization
scale, are both deferred to the normalize step (spec step 3) where a
single per-vector norm absorbs them. A rotation that is only correct up
to a scale factor is still a correct rotation for quantization
purposes, and keeping the scale out of the datapath keeps the golden
exact.

Emits into tb/:
    kv_x_k.memh     128 x INT8    real post-RoPE K, head 3
    kv_x_v.memh     128 x INT8    real V, head 3
    kv_had_k.memh   128 x 24-bit  Hadamard golden for K
    kv_had_v.memh   128 x 24-bit  Hadamard golden for V
    kv_pi_k.memh    128 x 24-bit  dense-Pi golden for K
    kv_pi_v.memh    128 x 24-bit  dense-Pi golden for V
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "rtl")
SOFT = os.path.join(os.path.dirname(HERE), "software")
sys.path.insert(0, SOFT)

from fetch_qwen import CACHE, fetch_embed_rows, fetch_file, load_header  # noqa: E402
import attention_l0 as A                                                # noqa: E402
from gen_kv_rot_tables import N, hadamard                               # noqa: E402

POSITION = 137     # same real mid-sequence position tb_rope uses
HEAD = 3           # fixed BEFORE looking at the per-head statistics


def crest(v):
    """peak-to-rms. The single number the rotation exists to reduce."""
    v = np.asarray(v, dtype=np.float64)
    r = np.sqrt((v ** 2).mean())
    return float(np.abs(v).max() / r) if r > 0 else 0.0


def w(name, vals, width):
    mask = (1 << width) - 1
    with open(os.path.join(HERE, name), "w") as f:
        f.write(chr(10).join("%0*x" % (width // 4, int(v) & mask)
                             for v in vals) + chr(10))


def load_real_kv():
    """Real layer-0 K (post-k_norm, post-RoPE) and V (raw), INT8.

    Shared with gen_kv_codebooks.py so the codebooks are fitted on
    exactly the vectors the testbench then quantizes - a codebook fitted
    on a different distribution than the one measured would make the
    step-5 numbers meaningless.
    """
    cfg, _, _, q = _build()
    return cfg, q


def _build():
    cfg = json.load(open(os.path.join(CACHE, "config.json")))
    eps = cfg["rms_norm_eps"]
    hd, base = cfg["head_dim"], cfg["rope_theta"]
    nkv, nf = cfg["num_key_value_heads"], cfg["head_dim"] // 2
    assert hd == N, "head_dim %d != rotation size %d" % (hd, N)

    from tokenizers import Tokenizer
    from quantize_pack import CALIB_TEXTS
    tok = Tokenizer.from_file(fetch_file("tokenizer.json"))
    seqs = [tok.encode(t).ids for t in CALIB_TEXTS]
    hdr, ds = load_header()
    hdr.pop("__metadata__", None)
    rows = fetch_embed_rows(hdr, ds, [i for s in seqs for i in s])

    g_in = A.T("model.layers.0.input_layernorm.weight")
    k_norm = A.T("model.layers.0.self_attn.k_norm.weight")
    Wk = A.T("model.layers.0.self_attn.k_proj.weight")
    Wv = A.T("model.layers.0.self_attn.v_proj.weight")

    # ---- real layer-0 K (post k_norm, post RoPE) and V (raw) --------
    allk, allv = [], []
    for sq in seqs:
        x = np.stack([rows[int(t)] for t in sq]).astype(np.float64)
        h = A.rmsnorm(x, g_in, eps)
        k = A.rmsnorm((h @ Wk.T).reshape(len(sq), nkv, hd), k_norm, eps)
        allk.append(k.reshape(-1, hd))
        allv.append((h @ Wv.T).reshape(-1, hd))       # V is NOT normed
    allk = np.concatenate(allk)
    allv = np.concatenate(allv)

    # RoPE, at the same position and convention tb_rope verified
    th = np.array([base ** (-2.0 * j / hd) for j in range(nf)])
    c, s_ = np.cos(POSITION * th), np.sin(POSITION * th)
    k1, k2 = allk[:, :nf], allk[:, nf:]
    allk = np.concatenate([k1 * c - k2 * s_, k2 * c + k1 * s_], axis=1)

    scales, quant = {}, {}
    for name, arr in (("k", allk), ("v", allv)):
        scales[name] = float(np.abs(arr).max()) / 127.0
        quant[name] = np.clip(np.rint(arr / scales[name]),
                              -128, 127).astype(np.int64)
    return cfg, nkv, scales, quant


def rot_tables():
    """The tables as the RTL will actually read them, re-decoded from
    the emitted memh rather than recomputed - so a table the hardware
    cannot load is a table the golden does not silently substitute."""
    sign = np.array([int(l, 16) for l in
                     open(os.path.join(RTL, "kv_rot_sign.memh"))], dtype=np.int64)
    d_vec = np.where(sign == 1, -1, 1)
    pi_q = np.array([int(l, 16) for l in
                     open(os.path.join(RTL, "kv_rot_pi.memh"))], dtype=np.int64)
    pi_q = np.where(pi_q > 127, pi_q - 256, pi_q).reshape(N, N)
    return hadamard(N), d_vec, pi_q


def main():
    cfg, nkv, scales, quant = _build()
    h_mat, d_vec, pi_q = rot_tables()
    for name in ("k", "v"):
        emit(name, quant[name], scales[name], h_mat, d_vec, pi_q, nkv)


def emit(name, q, scale, h_mat, d_vec, pi_q, nkv):
    # Where HEAD sits among all heads, so the choice is auditable
    # rather than convenient. Heads are interleaved every nkv rows.
    per_head = [crest(q[hh::nkv].reshape(-1)) for hh in range(nkv)]
    rank = sorted(range(nkv), key=lambda i: -per_head[i]).index(HEAD) + 1

    x = q[HEAD]
    y_had = h_mat @ (d_vec * x)
    y_pi = pi_q @ x

    assert np.abs(y_had).max() < (1 << 23), "Hadamard golden exceeds 24-bit"
    assert np.abs(y_pi).max() < (1 << 23), "dense golden exceeds 24-bit"
    # Parseval, exactly, in integers - the property tb_kv_rotate re-checks
    assert int((y_had ** 2).sum()) == N * int((x ** 2).sum()), "H^T H != N*I"

    w("kv_x_%s.memh" % name, x, 8)
    w("kv_had_%s.memh" % name, y_had, 24)
    w("kv_pi_%s.memh" % name, y_pi, 24)

    print("--- %s ---" % name)
    print("  INT8 input     min %4d max %4d  (scale %.6e)"
          % (x.min(), x.max(), scale))
    print("  head %d crest   %.3f   rank %d of %d heads (%.3f..%.3f)"
          % (HEAD, per_head[HEAD], rank, nkv,
             min(per_head), max(per_head)))
    print("  OUTLIER FLATTENING - peak/rms, the point of the rotation:")
    print("      input           %.3f" % crest(x))
    print("      Hadamard out    %.3f" % crest(y_had))
    print("      dense Pi out    %.3f" % crest(y_pi))
    print("  golden range   Hadamard [%d, %d]   dense [%d, %d]"
          % (y_had.min(), y_had.max(), y_pi.min(), y_pi.max()))


if __name__ == "__main__":
    main()
