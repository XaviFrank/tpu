#!/usr/bin/env python3
"""Fetch just the Qwen3-0.6B tensors this pipeline needs, over HTTP range
requests, straight out of the remote safetensors file.

Why range requests rather than a normal download: model.safetensors is
~1.2 GB and this pass needs about 4 MB of it. safetensors' layout makes
that easy - an 8-byte little-endian header length, then a JSON header
giving every tensor's dtype/shape/byte-range, then the raw data. So we
read the header, look up the byte range for the tensors we want, and
fetch only those.

Deliberately no torch and no transformers. The only third-party
dependencies are numpy and (for calibration) tokenizers, which keeps
this runnable without a multi-GB install. See README.md for what that
costs us at layer 1+.
"""

import json
import os
import struct
import sys
import urllib.request

import numpy as np

REPO = "Qwen/Qwen3-0.6B"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "qwen")

# Everything layer 0's Q-proj needs, and nothing else.
WANTED = [
    "model.layers.0.self_attn.q_proj.weight",     # [2048, 1024] the tensor under test
    "model.layers.0.input_layernorm.weight",      # [1024] RMSNorm gamma, feeds the activation
]


def _get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def bf16_to_f32(raw, count):
    """bfloat16 is simply the top 16 bits of a float32, so widening is a
    shift - no lookup table, no torch."""
    u16 = np.frombuffer(raw, dtype="<u2", count=count)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_header():
    """Return (header_dict, data_section_start_offset)."""
    os.makedirs(CACHE, exist_ok=True)
    hpath = os.path.join(CACHE, "header.json")
    lpath = os.path.join(CACHE, "hdr_len.txt")
    url = f"{BASE}/model.safetensors"

    if os.path.exists(hpath) and os.path.exists(lpath):
        n = int(open(lpath).read().strip())
    else:
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        open(lpath, "w").write(str(n))
        with open(hpath, "wb") as f:
            f.write(_get(url, 8, 8 + n - 1))

    return json.load(open(hpath)), 8 + n


def verify_dims(header):
    """Spec section 2: confirm the multiple-of-16 claim against the REAL
    file, not against arithmetic quoted from memory."""
    kinds = ("q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj")
    mm = {k: v for k, v in header.items()
          if k.endswith(".weight") and any(s in k for s in kinds)}
    ragged = {k: v["shape"] for k, v in mm.items()
              if any(d % 16 for d in v["shape"])}
    shapes = {}
    for v in mm.values():
        shapes[tuple(v["shape"])] = shapes.get(tuple(v["shape"]), 0) + 1

    print(f"  matmul weight tensors : {len(mm)}")
    print(f"  distinct shapes       : "
          f"{sorted((list(s), n) for s, n in shapes.items())}")
    print(f"  dims not mult. of 16  : {len(ragged)}")
    if ragged:
        for k, s in list(ragged.items())[:5]:
            print(f"      {k} {s}")
    return len(mm), len(ragged) == 0


def fetch_tensor(header, data_start, name):
    """Range-fetch one tensor and return it as float32."""
    e = header[name]
    assert e["dtype"] == "BF16", f"{name}: unexpected dtype {e['dtype']}"
    lo, hi = e["data_offsets"]
    raw = _get(f"{BASE}/model.safetensors", data_start + lo, data_start + hi - 1)
    n = int(np.prod(e["shape"]))
    return bf16_to_f32(raw, n).reshape(e["shape"])


def fetch_embed_rows(header, data_start, token_ids):
    """Fetch only the embedding rows for the given tokens.

    embed_tokens is [151936, 1024] BF16 = ~311 MB; a calibration set
    touches a few hundred rows at 2 KB each, so per-row ranges are far
    cheaper than pulling the table.
    """
    e = header["model.embed_tokens.weight"]
    vocab, hidden = e["shape"]
    lo = e["data_offsets"][0]
    row_bytes = hidden * 2
    out = {}
    for t in sorted(set(int(t) for t in token_ids)):
        assert 0 <= t < vocab, f"token id {t} out of range"
        s = data_start + lo + t * row_bytes
        out[t] = bf16_to_f32(_get(f"{BASE}/model.safetensors", s, s + row_bytes - 1),
                             hidden)
    return out


def fetch_file(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "wb") as f:
            f.write(_get(f"{BASE}/{name}"))
    return path


def main():
    os.makedirs(CACHE, exist_ok=True)
    print(f"repo: {REPO}")

    cfg = json.load(open(fetch_file("config.json")))
    print("\nconfig.json (spec section 2 check):")
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "rms_norm_eps", "tie_word_embeddings"):
        print(f"  {k:22s} {cfg[k]}")

    header, data_start = load_header()
    header.pop("__metadata__", None)
    print("\nshape audit over the real safetensors header:")
    n_mm, ok = verify_dims(header)
    print(f"  => all {n_mm} matmul tensors tile cleanly at N=16: {ok}")
    if not ok:
        print("  !! ragged tiles present - the packer would need padding")
        sys.exit(1)

    for name in WANTED:
        t = fetch_tensor(header, data_start, name)
        out = os.path.join(CACHE, name.replace("/", "_") + ".npy")
        np.save(out, t)
        print(f"\nsaved {os.path.basename(out)}  shape={t.shape} dtype={t.dtype}")
        print(f"  absmax={np.abs(t).max():.6f}  mean={t.mean():+.6e}  std={t.std():.6e}")

    fetch_file("tokenizer.json")
    print("\ntokenizer.json cached")
    print("done.")


if __name__ == "__main__":
    main()
