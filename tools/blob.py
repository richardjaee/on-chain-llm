"""Read the on-chain blob layout back into the arrays the reference forward pass wants.

`convert.py` writes the blob and the Solidity reads it; until now nothing in Python read
it back, because the reference was always handed its weights from the checkpoint directly.
A synthetic mixture has no checkpoint, so the blob is the only source of truth and
something has to parse it.

Offsets are derived from the config, exactly as `Transformer.layout` derives them. That
duplication is the point: if the two disagree, the reference and the contract are reading
different models and every comparison between them is meaningless. Two independent
derivations that agree is evidence; one shared helper would hide a mistake in both.
"""

import struct


def block_size(rows, cols):
    """One tensor block: `rows` shift bytes, then `rows * cols` int16."""
    return rows + rows * cols * 2


def layout(cfg):
    """Byte offset of every tensor's shift table, in the order the blob stores them."""
    dim, hid, L, V = cfg["dim"], cfg["hidden"], cfg["n_layers"], cfg["vocab"]
    kvd, hs = cfg["kvd"], cfg["hs"]
    E = cfg.get("n_experts") or 1

    o, at = {}, 0

    def place(name, rows, cols):
        nonlocal at
        o[name] = (at, rows, cols)
        at += block_size(rows, cols)

    place("tok_emb", V, dim)
    place("rms_att", 1, L * dim)
    place("wq", L * dim, dim)
    place("wk", L * kvd, dim)
    place("wv", L * kvd, dim)
    place("wo", L * dim, dim)
    place("rms_ffn", 1, L * dim)
    # The router sits between the norm and the experts, and only when there are experts —
    # a dense blob has no router block at all, which is what keeps its offsets unchanged.
    if E > 1:
        place("router", L * E, dim)
    place("w1", L * E * hid, dim)
    place("w2", L * E * dim, hid)
    place("w3", L * E * hid, dim)
    place("rms_final", 1, dim)
    if not cfg.get("shared", True):
        place("wcls", V, dim)
    o["rope_cos"] = (at, cfg["seqlen"], hs // 2)
    at += cfg["seqlen"] * (hs // 2) * 4
    o["rope_sin"] = (at, cfg["seqlen"], hs // 2)
    at += cfg["seqlen"] * (hs // 2) * 4
    o["__len__"] = at
    return o


def read(blob, cfg):
    """Parse into `(weights, shifts, cos, sin)` keyed the way `model.Model` expects.

    A shared classifier reuses the embedding block rather than getting its own, so `wcls`
    aliases `tok_emb` instead of being copied — the same aliasing the contract does.
    """
    o = layout(cfg)
    if len(blob) != o["__len__"]:
        raise ValueError(f"blob is {len(blob)} bytes, layout derives {o['__len__']}")

    weights, shifts = {}, {}
    for name, entry in o.items():
        # `__len__` is a bare int and the rope tables are read separately below, so both
        # are filtered before the entry is destructured rather than after.
        if name.startswith("__") or name.startswith("rope_"):
            continue
        at, rows, cols = entry
        shifts[name] = list(blob[at : at + rows])
        vals = blob[at + rows : at + rows + rows * cols * 2]
        weights[name] = list(struct.unpack(f">{rows * cols}h", vals))

    if cfg.get("shared", True):
        weights["wcls"], shifts["wcls"] = weights["tok_emb"], shifts["tok_emb"]

    def table(key):
        at, n, half = o[key]
        flat = struct.unpack(f">{n * half}i", blob[at : at + n * half * 4])
        return [list(flat[p * half : (p + 1) * half]) for p in range(n)]

    return weights, shifts, table("rope_cos"), table("rope_sin")


def cfg_from(dim, hidden, n_layers, vocab, heads, kv_heads, seqlen, n_experts=0, top_k=0):
    """The config `synth.py` was invoked with, in the shape `Model` and `layout` want."""
    hs = dim // heads
    return {
        "dim": dim,
        "hidden": hidden,
        "n_layers": n_layers,
        "vocab": vocab,
        "n_heads": heads,
        "n_kv": kv_heads,
        "hs": hs,
        "kvd": hs * kv_heads,
        "seqlen": seqlen,
        "shared": True,
        "n_experts": n_experts,
        "top_k": top_k,
    }
