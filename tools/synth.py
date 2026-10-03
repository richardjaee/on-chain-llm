"""Generate a synthetic model blob in the on-chain layout, for scaling experiments.

Gas in the EVM does not depend on the *values* being multiplied, only on the shapes
and the memory and storage touched. So measuring how a wide model behaves does not
require a trained one — it requires a correctly shaped one. This writes a blob with
the same layout `convert.py` produces, filled with deterministic pseudo-random bytes,
so the contracts can be exercised at widths no real checkpoint here reaches.

    python3 tools/synth.py --dim 256 --hidden 688 --layers 2 --vocab 512 \
        --heads 8 --kv-heads 4 --seqlen 256 --out artifacts/synth.hex
"""
import argparse, hashlib, struct


def prng(n, seed):
    """Deterministic bytes, cheaply: repeated sha256 of a counter."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


def block(rows, cols, seed):
    """One tensor block: [u8 shift per row][int16 values, row-major]."""
    # Shifts in a plausible range; the reference exponents span 14-18 on the real
    # checkpoint, so keep them there rather than uniform over a byte.
    shifts = bytes((14 + (b % 5)) for b in prng(rows, f"s{seed}"))
    return shifts + prng(rows * cols * 2, f"v{seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--seqlen", type=int, required=True)
    p.add_argument(
        "--experts",
        type=int,
        default=0,
        help="experts per layer; 0 or 1 is dense and lays out byte for byte as before",
    )
    p.add_argument("--top-k", type=int, default=0, help="experts a token visits")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dim, hid, L, V = a.dim, a.hidden, a.layers, a.vocab
    hs = dim // a.heads
    kvd = hs * a.kv_heads
    assert dim % a.heads == 0 and a.heads % a.kv_heads == 0, "head geometry must divide"

    # One expert is the dense case rather than a special case of it, so every size below
    # multiplies by E and a dense blob comes out byte for byte as it did before there was
    # an expert axis. Transformer.sol derives its offsets the same way; LayoutTest pins it.
    E = a.experts if a.experts else 1
    if E > 1:
        assert 1 <= a.top_k <= E, "top-k must be between one and the expert count"

    parts = []
    parts.append(block(V, dim, "tok"))            # tok_emb
    parts.append(block(1, L * dim, "rmsa"))       # rms_att
    parts.append(block(L * dim, dim, "wq"))
    parts.append(block(L * kvd, dim, "wk"))
    parts.append(block(L * kvd, dim, "wv"))
    parts.append(block(L * dim, dim, "wo"))
    parts.append(block(1, L * dim, "rmsf"))       # rms_ffn
    if E > 1:
        # The router sits between the norm and the experts, one score row per expert.
        parts.append(block(L * E, dim, "router"))
    parts.append(block(L * E * hid, dim, "w1"))
    parts.append(block(L * E * dim, hid, "w2"))
    parts.append(block(L * E * hid, dim, "w3"))
    parts.append(block(1, dim, "rmsfin"))         # rms_final (classifier shares tok_emb)

    # RoPE tables, Q30 cos/sin per position and pair
    half = hs // 2
    rope = bytearray()
    for table in ("cos", "sin"):
        for pos in range(a.seqlen):
            for q in range(half):
                # Values are irrelevant to gas; keep them in Q30 range so nothing
                # overflows if the blob is ever run for real.
                v = int.from_bytes(prng(4, f"r{table}{pos}{q}"), "big") % (1 << 30)
                rope += struct.pack(">i", v)
    parts.append(bytes(rope))

    blob = b"".join(parts)
    with open(a.out, "w") as f:
        f.write(blob.hex())

    params = V * dim + L * (dim * dim + 2 * kvd * dim + dim * dim + 3 * hid * dim)
    print(f"dim={dim} hidden={hid} layers={L} vocab={V} heads={a.heads} kv={a.kv_heads}")
    print(f"  hs={hs} kvd={kvd}")
    print(f"  parameters : {params/1e6:.2f}M")
    print(f"  blob bytes : {len(blob):,} ({len(blob)/1e6:.2f} MB)")
    print(f"  chunks     : {(len(blob) + 24574)//24575}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
