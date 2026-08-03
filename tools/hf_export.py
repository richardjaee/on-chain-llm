"""Export a HuggingFace Llama-architecture model to llama2.c's legacy format.

`convert.py` reads llama2.c's export: a 7 x int32 header followed by fp32 tensors in a
fixed order. HuggingFace ships safetensors with named tensors and, more importantly, a
different rotary convention. This bridges the two, which is all that stands between the
contracts and any model whose config says:

    architectures    ["LlamaForCausalLM"]
    attention_bias   false          the forward pass has no bias terms anywhere
    mlp_bias         false
    hidden_act       "silu"         SwiGLU, not GeGLU

THE ROTARY CONVENTION IS THE WHOLE PROBLEM. llama2.c rotates adjacent pairs
(x0,x1), (x2,x3), ... HuggingFace splits the head in half and rotates (x_i, x_{i+hs/2}).
The two are related by a permutation of the rows of W_q and W_k, so the weights must be
un-permuted on the way out. Get this wrong and nothing errors: the model simply produces
confident nonsense, because every value is still in range and only the pairing is wrong.

    python3 tools/hf_export.py --model path/to/hf_dir --out model.bin
"""
import argparse, json, struct
import numpy as np


def read_safetensors(path):
    """Parse the container directly: [u64 header length][JSON header][raw tensor data]."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        raw = blob[s:e]
        shape = meta["shape"]
        dt = meta["dtype"]
        if dt == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        elif dt == "F16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dt == "BF16":
            # bfloat16 is the top 16 bits of a float32, so widening is a shift
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            a = u.view(np.float32)
        else:
            raise SystemExit(f"unsupported dtype {dt} for {name}")
        out[name] = a.reshape(shape)
    return out


def unpermute(w, n_heads, hs):
    """HuggingFace rotary order -> llama2.c interleaved order.

    The HF conversion reshapes each head's rows as (hs/2, 2) and transposes; this undoes
    it by reshaping as (2, hs/2) and transposing back.
    """
    rows, cols = w.shape
    return w.reshape(n_heads, 2, hs // 2, cols).transpose(0, 2, 1, 3).reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="directory with config.json and model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlen", type=int, help="cap the position table (default: the model's)")
    a = ap.parse_args()

    cfg = json.load(open(f"{a.model}/config.json"))
    for field, want in [("attention_bias", False), ("mlp_bias", False), ("hidden_act", "silu")]:
        if cfg.get(field, want) != want:
            raise SystemExit(f"config {field}={cfg.get(field)!r}, the contracts require {want!r}")
    if cfg["architectures"] != ["LlamaForCausalLM"]:
        raise SystemExit(f"architecture {cfg['architectures']} is not LlamaForCausalLM")

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    hs = cfg.get("head_dim", dim // H)
    seqlen = a.seqlen or cfg["max_position_embeddings"]
    tied = cfg.get("tie_word_embeddings", False)
    theta = cfg.get("rope_theta", 10000.0)

    if dim % H or H % KV or hs % 2:
        raise SystemExit("head geometry must divide, and the head size must be even for RoPE")

    t = read_safetensors(f"{a.model}/model.safetensors")
    get = lambda k: t[k] if k in t else t["model." + k]

    def layers(fmt):
        return np.concatenate([get(fmt.format(l=l)).reshape(-1) for l in range(L)])

    parts = []
    parts.append(get("embed_tokens.weight").reshape(-1))
    parts.append(layers("layers.{l}.input_layernorm.weight"))
    parts.append(np.concatenate([
        unpermute(get(f"layers.{l}.self_attn.q_proj.weight"), H, hs).reshape(-1) for l in range(L)]))
    parts.append(np.concatenate([
        unpermute(get(f"layers.{l}.self_attn.k_proj.weight"), KV, hs).reshape(-1) for l in range(L)]))
    parts.append(layers("layers.{l}.self_attn.v_proj.weight"))
    parts.append(layers("layers.{l}.self_attn.o_proj.weight"))
    parts.append(layers("layers.{l}.post_attention_layernorm.weight"))
    parts.append(layers("layers.{l}.mlp.gate_proj.weight"))
    parts.append(layers("layers.{l}.mlp.down_proj.weight"))
    parts.append(layers("layers.{l}.mlp.up_proj.weight"))
    parts.append(get("norm.weight").reshape(-1))

    # Legacy freq_cis tables. llama2.c ignores them and convert.py skips them, but the
    # reader walks the file by offset so they have to occupy their space.
    parts.append(np.zeros(seqlen * (hs // 2) * 2, dtype=np.float32))

    if not tied:
        parts.append(t["lm_head.weight"].reshape(-1))

    header = struct.pack("<7i", dim, hidden, L, H, KV, V if tied else -V, seqlen)
    body = np.concatenate([p.astype(np.float32) for p in parts])
    with open(a.out, "wb") as f:
        f.write(header)
        f.write(body.tobytes())

    kvd = hs * KV
    params = V * dim + L * (dim * dim + 2 * kvd * dim + dim * dim + 3 * hidden * dim) + (0 if tied else V * dim)
    print(f"dim={dim} hidden={hidden} layers={L} heads={H} kv={KV} hs={hs} vocab={V}")
    print(f"  seqlen {seqlen}, rope_theta {theta}, tied classifier {tied}")
    print(f"  parameters {params/1e6:.2f}M")
    print(f"  wrote {a.out} ({(28 + body.nbytes)/1e6:.1f} MB)")
    if theta != 10000.0:
        print(f"  NOTE: pass --rope-theta {theta} to convert.py")


if __name__ == "__main__":
    main()
