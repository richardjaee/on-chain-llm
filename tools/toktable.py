"""Build the on-chain token table.

Layout (ours), all big-endian:
    [u16 count][u8 maxLen][u32 offset per token][concatenated token bytes]

Offsets are 32-bit. A 512-token vocabulary needs ~2KB of strings and fits in 16 bits,
but Llama's 32k vocabulary needs ~178KB and does not — with 16-bit offsets it decodes
to nonsense rather than failing, which is the worst way for it to be wrong.

Two sources:
    --tokenizer tokenizer.bin        llama2.c's format: [i32 maxLen] then per token
                                     [f32 score][i32 len][bytes]
    --jsspec frontend/_tok_inline.js the tok512 table embedded in the console, whose
                                     entries are [score, hex] in id order
    --hf tokenizer.json             a HuggingFace byte-level BPE vocabulary

    python3 tools/toktable.py --tokenizer tokenizer.bin --out artifacts/tokens15m.hex
"""
import argparse, json, re, struct


def from_tokenizer_bin(path):
    d = open(path, "rb").read()
    off = 4  # max_token_length, recomputed below
    toks = []
    while off < len(d):
        off += 4  # score, unused for decoding
        (ln,) = struct.unpack("<i", d[off:off + 4])
        off += 4
        toks.append(d[off:off + ln])
        off += ln
    return toks


def from_hf_tokenizer(path):
    """A HuggingFace tokenizer.json using GPT-2 byte-level BPE.

    Those vocabularies do not store the bytes a token stands for; they store each byte
    remapped to a printable character, so a leading space is '\u0120' rather than ' '.
    The mapping is inverted here, at build time, so the table on chain holds the actual
    bytes and `Decoder.sol` needs to know nothing about the convention. Doing it the
    other way — teaching the contract the 256-entry mapping — would cost storage and gas
    on every decode to represent something already known off chain.
    """
    tj = json.load(open(path))
    vocab = tj["model"]["vocab"]
    # the GPT-2 byte<->unicode table
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}

    toks = [b""] * len(vocab)
    for tok, i in vocab.items():
        try:
            toks[i] = bytes(u2b[ch] for ch in tok)
        except KeyError:
            # added/control tokens such as <|im_start|> are literal, not byte-mapped
            toks[i] = tok.encode("utf-8")
    return toks


def from_js_spec(path):
    src = open(path).read()
    m = re.search(r"const TOK512_SPEC = (\{.*?\});", src, re.S)
    if not m:
        raise SystemExit("TOK512_SPEC not found in " + path)
    spec = json.loads(m.group(1))
    return [bytes.fromhex(h) for _, h in spec["tokens"]]


def build(toks):
    n = len(toks)
    assert n <= 0xFFFF, f"{n} tokens exceeds the u16 count field"
    max_len = max(len(t) for t in toks)
    assert max_len <= 0xFF, f"token of {max_len} bytes exceeds the u8 maxLen field"
    out = bytearray()
    out += struct.pack(">H", n)
    out += bytes([max_len])
    at = 0
    for t in toks:
        out += struct.pack(">I", at)
        at += len(t)
    assert at <= 0xFFFFFFFF, "string region exceeds a 32-bit offset"
    for t in toks:
        out += t
    return bytes(out)


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokenizer", help="llama2.c tokenizer.bin")
    src.add_argument("--jsspec", help="frontend/_tok_inline.js, for tok512")
    src.add_argument("--hf", help="HuggingFace tokenizer.json (GPT-2 byte-level BPE)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, help="truncate to this many tokens")
    a = p.parse_args()

    if a.tokenizer:
        toks = from_tokenizer_bin(a.tokenizer)
    elif a.hf:
        toks = from_hf_tokenizer(a.hf)
    else:
        toks = from_js_spec(a.jsspec)
    if a.limit:
        toks = toks[:a.limit]
    table = build(toks)
    with open(a.out, "w") as f:
        f.write(table.hex())
    print(f"tokens {len(toks)}, strings {sum(len(t) for t in toks):,} bytes, "
          f"table {len(table):,} bytes -> {a.out}")


if __name__ == "__main__":
    main()
