#!/usr/bin/env python3
"""Export a Mixtral-architecture mixture checkpoint into the on-chain blob layout.

`convert.py` reads llama2.c's format, which has no notion of experts, so a mixture cannot
reach the chain through it no matter what is added to either end. This reads the
HuggingFace checkpoint directly.

Mixtral rather than Qwen3-MoE, and that is a constraint rather than a preference.
`Transformer.sol` implements llama2's attention — RMSNorm, RoPE, grouped-query attention,
SwiGLU — and Qwen3 adds a per-head norm on queries and keys that is not there. Mixtral is
llama2 with experts, so it runs as-is. A Qwen3 mixture would need the contract changed
first, and the companion project is where that already exists.

    python3 tools/export_mixtral.py --model <dir> --out artifacts/mixtral.hex \\
        --meta artifacts/mixtral.json

The quantiser and the layout are both imported rather than reimplemented. A second copy of
either would be a second thing to keep in step with the contract.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert import quantize_rows  # noqa: E402
from hf_export import read_safetensors  # noqa: E402


def block(vals, rows, cols):
    """One tensor block: [u8 shift per row][int16 values, row-major, big-endian]."""
    q, shifts = quantize_rows(vals, rows, cols)
    assert len(shifts) == rows, f"{len(shifts)} shifts for {rows} rows"
    return bytes(shifts) + np.asarray(q, dtype=">i2").tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="directory with config.json and model.safetensors")
    ap.add_argument("--out", required=True, help="blob, as hex")
    ap.add_argument("--meta", required=True, help="config the contract is constructed from")
    ap.add_argument("--positions", type=int, default=8, help="rotary table length")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    dim = cfg["hidden_size"]
    hid = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    E = cfg["num_local_experts"]
    K = cfg["num_experts_per_tok"]
    # `head_dim` is null in the checkpoints that predate the field; the contract derives
    # it the same way, so deriving it here keeps one definition rather than two.
    hs = cfg.get("head_dim") or dim // H
    kvd = hs * KV
    theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])

    assert dim % H == 0 and H % KV == 0, "head geometry must divide"
    assert 1 <= K <= E, "experts per token must be between one and the expert count"

    print(f"Mixtral: dim {dim}, hidden {hid}, layers {L}, heads {H}, kv {KV}, "
          f"head_dim {hs}, vocab {V}, experts {E}, top-k {K}")

    t = read_safetensors(os.path.join(a.model, "model.safetensors"))
    g = lambda n: t[n].astype(np.float64)
    shared = "lm_head.weight" not in t

    def layers(name):
        """One tensor per layer, stacked, which is how the blob orders them."""
        return np.concatenate([g(f"model.layers.{l}.{name}").reshape(-1) for l in range(L)])

    def experts(name, per_expert):
        """One tensor per (layer, expert), stacked in that order.

        `expertRow` in the contract is `(layer * experts + expert) * rowsEach`, so layer 0's
        experts come before layer 1's. Getting this order wrong reads another expert's
        weights, which is a wrong answer rather than an error.
        """
        out = []
        for l in range(L):
            w = g(f"model.layers.{l}.mlp.experts.{name}")
            for e in range(E):
                out.append(per_expert(w, e).reshape(-1))
        return np.concatenate(out)

    parts = []
    parts.append(block(g("model.embed_tokens.weight").reshape(-1), V, dim))
    parts.append(block(layers("input_layernorm.weight"), 1, L * dim))
    parts.append(block(layers("self_attn.q_proj.weight"), L * dim, dim))
    parts.append(block(layers("self_attn.k_proj.weight"), L * kvd, dim))
    parts.append(block(layers("self_attn.v_proj.weight"), L * kvd, dim))
    parts.append(block(layers("self_attn.o_proj.weight"), L * dim, dim))
    parts.append(block(layers("post_attention_layernorm.weight"), 1, L * dim))

    # The router, one score row per expert. `mlp.gate` is HuggingFace's name for it, which
    # collides with the feed-forward's gate projection; they are different tensors.
    parts.append(block(layers("mlp.gate.weight"), L * E, dim))

    # `gate_up_proj` is [experts, 2 * hidden, dim], gate first and up second. That is the
    # documented default, and it is an assumption here rather than something checked: the
    # two halves are shape-identical, so swapping them computes up * silu(gate) instead of
    # silu(gate) * up and every structural check still passes. Verifying it needs a
    # reference forward pass from transformers, which this repository does not have.
    #
    # The shape is checked, which catches a wrong intermediate size but not a swap.
    for l in range(L):
        got = t[f"model.layers.{l}.mlp.experts.gate_up_proj"].shape
        assert tuple(got) == (E, 2 * hid, dim), f"layer {l}: gate_up_proj is {got}, expected {(E, 2 * hid, dim)}"
        got = t[f"model.layers.{l}.mlp.experts.down_proj"].shape
        assert tuple(got) == (E, dim, hid), f"layer {l}: down_proj is {got}, expected {(E, dim, hid)}"

    parts.append(block(experts("gate_up_proj", lambda w, e: w[e][:hid]), L * E * hid, dim))
    parts.append(block(experts("down_proj", lambda w, e: w[e]), L * E * dim, hid))
    parts.append(block(experts("gate_up_proj", lambda w, e: w[e][hid:]), L * E * hid, dim))

    parts.append(block(g("model.norm.weight").reshape(-1), 1, dim))
    if not shared:
        parts.append(block(g("lm_head.weight").reshape(-1), V, dim))

    # Rotary tables, Q30, the convention the contract reads.
    half = hs // 2
    rope = bytearray()
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    ang = np.arange(a.positions, dtype=np.float64)[:, None] * inv[None, :]
    for table in (np.cos(ang), np.sin(ang)):
        for row in table:
            for v in row:
                rope += struct.pack(">i", int(round(v * (1 << 30))))
    parts.append(bytes(rope))

    blob = b"".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(blob.hex())

    meta = {
        "dim": dim, "hidden": hid, "n_layers": L, "n_heads": H, "n_kv": KV,
        "vocab": V, "seqlen": a.positions, "hs": hs, "kvd": kvd,
        "shared": shared, "n_experts": E, "top_k": K, "rope_theta": theta,
    }
    with open(a.meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wrote {a.out} ({len(blob)} bytes)")
    print(f"  wrote {a.meta}")


if __name__ == "__main__":
    main()
