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
    missing = [
        n for n in (f"model.layers.0.block_sparse_moe.experts.0.{w}.weight" for w in ("w1", "w2", "w3"))
        if n not in t
    ]
    if missing:
        raise SystemExit(
            "checkpoint does not use the block_sparse_moe naming this reads: missing "
            f"{missing[0]}. A checkpoint storing experts fused as mlp.experts.gate_up_proj "
            "needs splitting first, and the two halves are shape-identical, so that split "
            "cannot be inferred from the file."
        )
    g = lambda n: t[n].astype(np.float64)
    shared = "lm_head.weight" not in t

    def layers(name):
        """One tensor per layer, stacked, which is how the blob orders them."""
        return np.concatenate([g(f"model.layers.{l}.{name}").reshape(-1) for l in range(L)])

    def experts(which):
        """One expert projection per (layer, expert), stacked in that order.

        `expertRow` in the contract is `(layer * experts + expert) * rowsEach`, so layer 0's
        experts come before layer 1's. Getting this order wrong reads another expert's
        weights, which is a wrong answer rather than an error.

        `which` is `w1`, `w2` or `w3`, the names transformers gives these: gate, down and
        up respectively. Using them rather than a fused tensor is what removes the only
        ambiguity here — a fused `gate_up_proj` has two shape-identical halves, so reading
        them in the wrong order computes `up * silu(gate)` and every structural check still
        passes. Named projections cannot be swapped silently.
        """
        return np.concatenate([
            g(f"model.layers.{l}.block_sparse_moe.experts.{e}.{which}.weight").reshape(-1)
            for l in range(L)
            for e in range(E)
        ])

    parts = []
    parts.append(block(g("model.embed_tokens.weight").reshape(-1), V, dim))
    parts.append(block(layers("input_layernorm.weight"), 1, L * dim))
    parts.append(block(layers("self_attn.q_proj.weight"), L * dim, dim))
    parts.append(block(layers("self_attn.k_proj.weight"), L * kvd, dim))
    parts.append(block(layers("self_attn.v_proj.weight"), L * kvd, dim))
    parts.append(block(layers("self_attn.o_proj.weight"), L * dim, dim))
    parts.append(block(layers("post_attention_layernorm.weight"), 1, L * dim))

    # The router, one score row per expert.
    parts.append(block(layers("block_sparse_moe.gate.weight"), L * E, dim))

    parts.append(block(experts("w1"), L * E * hid, dim))   # gate
    parts.append(block(experts("w2"), L * E * dim, hid))   # down
    parts.append(block(experts("w3"), L * E * hid, dim))   # up

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
