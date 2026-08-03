"""Convert a llama2.c checkpoint into integer artifacts for on-chain inference.

The input format is llama2.c's own legacy export, read exactly as `read_checkpoint`
and `memory_map_weights` do in run.c (MIT): a 7 x int32 header followed by fp32
tensors, with a negative vocab_size signalling an unshared classifier.

Quantisation is ours: int16 with a PER-ROW power-of-two scale. One exponent per
matrix (the usual choice) is set by the largest-magnitude row and throws away
precision on every quieter row; a per-row exponent costs one byte per row and
recovers those bits. Power-of-two scales keep dequantisation a shift rather than
a multiply-divide.
"""
import argparse, json, math, struct

TENSOR_2D = ["tok_emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3"]
TENSOR_1D = ["rms_att", "rms_ffn", "rms_final"]


def read_checkpoint(path):
    with open(path, "rb") as f:
        blob = f.read()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seqlen = struct.unpack("<7i", blob[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    hs = dim // n_heads
    kvd = hs * n_kv
    off = [28]

    def take(n):
        s = off[0]
        vals = struct.unpack(f"<{n}f", blob[s:s + 4 * n])
        off[0] = s + 4 * n
        return list(vals)

    w = {}
    w["tok_emb"] = take(vocab * dim)
    w["rms_att"] = take(n_layers * dim)
    w["wq"] = take(n_layers * dim * dim)
    w["wk"] = take(n_layers * dim * kvd)
    w["wv"] = take(n_layers * dim * kvd)
    w["wo"] = take(n_layers * dim * dim)
    w["rms_ffn"] = take(n_layers * dim)
    w["w1"] = take(n_layers * dim * hidden)
    w["w2"] = take(n_layers * hidden * dim)
    w["w3"] = take(n_layers * dim * hidden)
    w["rms_final"] = take(dim)
    take(seqlen * hs // 2)   # legacy freq_cis_real, unused: RoPE is recomputed
    take(seqlen * hs // 2)   # legacy freq_cis_imag, unused
    w["wcls"] = w["tok_emb"] if shared else take(vocab * dim)
    assert off[0] == len(blob), f"trailing bytes: {len(blob) - off[0]}"

    cfg = dict(dim=dim, hidden=hidden, n_layers=n_layers, n_heads=n_heads, n_kv=n_kv,
               vocab=vocab, seqlen=seqlen, hs=hs, kvd=kvd, shared=shared)
    return cfg, w


def quantize_rows(vals, rows, cols, bits=16):
    """Per-row power-of-two scale. Returns (ints, shift-per-row)."""
    lim = (1 << (bits - 1)) - 1
    q = [0] * (rows * cols)
    shifts = []
    for r in range(rows):
        row = vals[r * cols:(r + 1) * cols]
        mx = max(abs(v) for v in row) if row else 0.0
        s = 0
        if mx > 0:
            while mx * (1 << (s + 1)) <= lim and s < 30:
                s += 1
        else:
            s = 30
        shifts.append(s)
        base = r * cols
        for j, v in enumerate(row):
            q[base + j] = min(lim, max(-lim - 1, round(v * (1 << s))))
    return q, shifts


def rope_tables(cfg):
    """cos/sin per position and pair, Q30. Recomputed rather than trusting the
    legacy tables, which llama2.c itself ignores."""
    hs = cfg["hs"]
    cos_t, sin_t = [], []
    for pos in range(cfg["seqlen"]):
        c, s = [], []
        for i in range(hs // 2):
            freq = 1.0 / (10000.0 ** (2 * i / hs))
            c.append(round(math.cos(pos * freq) * (1 << 30)))
            s.append(round(math.sin(pos * freq) * (1 << 30)))
        cos_t.append(c)
        sin_t.append(s)
    return cos_t, sin_t


def convert(path):
    cfg, w = read_checkpoint(path)
    dim, hidden, L = cfg["dim"], cfg["hidden"], cfg["n_layers"]
    kvd, vocab = cfg["kvd"], cfg["vocab"]

    shapes = {
        "tok_emb": (vocab, dim),
        "wq": (L * dim, dim), "wk": (L * kvd, dim), "wv": (L * kvd, dim),
        "wo": (L * dim, dim),
        "w1": (L * hidden, dim), "w2": (L * dim, hidden), "w3": (L * hidden, dim),
    }
    qw, shifts = {}, {}
    for name, (rows, cols) in shapes.items():
        qw[name], shifts[name] = quantize_rows(w[name], rows, cols)
    # 1-D norm weights: a single row, so one shared exponent
    for name in TENSOR_1D:
        n = len(w[name])
        q, s = quantize_rows(w[name], 1, n)
        qw[name], shifts[name] = q, s
    qw["wcls"], shifts["wcls"] = (qw["tok_emb"], shifts["tok_emb"]) if cfg["shared"] \
        else quantize_rows(w["wcls"], vocab, dim)

    cos_t, sin_t = rope_tables(cfg)
    return cfg, qw, shifts, cos_t, sin_t


# ---------------------------------------------------------------- blob layout
# Ours, and deliberately simple so the Solidity can derive every offset from the
# config alone. For each tensor, in this order:
#     [u8 shift per row][int16 values, row-major, big-endian]
# then the RoPE tables:
#     [int32 cos, seqlen x hs/2][int32 sin, seqlen x hs/2]
ORDER = ["tok_emb", "rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3", "rms_final"]


def tensor_shape(cfg, name):
    dim, hid, L = cfg["dim"], cfg["hidden"], cfg["n_layers"]
    return {
        "tok_emb": (cfg["vocab"], dim), "rms_att": (1, L * dim), "wq": (L * dim, dim),
        "wk": (L * cfg["kvd"], dim), "wv": (L * cfg["kvd"], dim), "wo": (L * dim, dim),
        "rms_ffn": (1, L * dim), "w1": (L * hid, dim), "w2": (L * dim, hid),
        "w3": (L * hid, dim), "rms_final": (1, dim), "wcls": (cfg["vocab"], dim),
    }[name]


def build_blob(cfg, qw, shifts, cos_t, sin_t):
    out = bytearray()
    for name in ORDER + ([] if cfg["shared"] else ["wcls"]):
        rows, cols = tensor_shape(cfg, name)
        sh = shifts[name]
        assert len(sh) == rows, f"{name}: {len(sh)} shifts for {rows} rows"
        out += bytes(sh)
        for v in qw[name]:
            out += struct.pack(">h", v)
    for row in cos_t:
        for v in row:
            out += struct.pack(">i", v)
    for row in sin_t:
        for v in row:
            out += struct.pack(">i", v)
    return bytes(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--blob", help="write the on-chain blob here as hex")
    a = ap.parse_args()
    cfg, qw, shifts, cos_t, sin_t = convert(a.model)
    json.dump({"cfg": cfg, "shifts": shifts}, open(a.out, "w"))
    print("config:", cfg)
    for k in sorted(shifts):
        s = shifts[k]
        print(f"  {k:10} rows={len(s):>6}  shift min/max = {min(s)}/{max(s)}")
    if a.blob:
        blob = build_blob(cfg, qw, shifts, cos_t, sin_t)
        with open(a.blob, "w") as f:
            f.write(blob.hex())
        print(f"blob: {len(blob):,} bytes ({len(blob)/1e6:.2f} MB), "
              f"{(len(blob) + 24574)//24575} chunks -> {a.blob}")
