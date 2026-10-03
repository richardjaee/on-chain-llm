#!/usr/bin/env python3
"""Generate a synthetic mixture and what the specification says it emits.

The dispatch in `OnchainLLM.sol` has compiled since it was written and has never run,
because nothing could produce a blob with an expert axis and nothing could say what the
answer should be. This produces both.

Synthetic weights are enough for what is being tested. Whether the router picks the
experts the specification picks, whether their rows are read from the right offsets, and
whether the phase ladder walks them in the right order are all properties of the dispatch
rather than of the values — and a wrong answer on random weights is still a wrong answer.

What it does not test is whether a real mixture produces good text. That needs a real
checkpoint and `convert.py`, and is a separate claim.

    python3 tools/moe_run.py --experts 4 --top-k 2 --out artifacts
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blob as blobmod  # noqa: E402
from model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Small, because the Solidity side runs this through the resumable contract a phase at
    # a time and a wide model would spend the test's gas on arithmetic already covered.
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    # Matches artifacts/tokens.hex, so the directory the contract expects can reuse it.
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=8)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--prompt", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--new", type=int, default=2, help="tokens to generate past the prompt")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    hexfile = os.path.join(a.out, "moe_synth.hex")
    os.makedirs(a.out, exist_ok=True)

    subprocess.run(
        [
            sys.executable, os.path.join(here, "synth.py"),
            "--dim", str(a.dim), "--hidden", str(a.hidden), "--layers", str(a.layers),
            "--vocab", str(a.vocab), "--heads", str(a.heads), "--kv-heads", str(a.kv_heads),
            "--seqlen", str(a.seqlen), "--experts", str(a.experts), "--top-k", str(a.top_k),
            "--out", hexfile,
        ],
        check=True,
        capture_output=True,
    )
    raw = bytes.fromhex(open(hexfile).read().strip())

    cfg = blobmod.cfg_from(
        a.dim, a.hidden, a.layers, a.vocab, a.heads, a.kv_heads, a.seqlen,
        n_experts=a.experts, top_k=a.top_k,
    )
    w, sh, cos, sin = blobmod.read(raw, cfg)
    m = Model(cfg, w, sh, cos, sin)

    # Generation, exactly as the contract does it: the prompt's next token is already
    # known, so only positions past it take the classifier's answer.
    total = len(a.prompt) + a.new
    assert total <= a.seqlen, "prompt plus new tokens must fit the rotary tables"
    st = m.new_state(total)
    tokens, logits0 = [], None
    tok = a.prompt[0]
    for pos in range(total - 1):
        lg = m.forward(st, tok, pos)
        if pos == 0:
            logits0 = lg
        if pos + 1 < len(a.prompt):
            tok = a.prompt[pos + 1]
        else:
            from intref import argmax
            tok = argmax(lg)
            tokens.append(tok)

    out = {
        "cfg": {
            "dim": a.dim, "hidden": a.hidden, "n_layers": a.layers, "vocab": a.vocab,
            "n_heads": a.heads, "n_kv": a.kv_heads, "seqlen": a.seqlen,
            "n_experts": a.experts, "top_k": a.top_k,
        },
        "prompt": a.prompt,
        "gen": tokens,
        # Quoted, because stdJson reads unsigned arrays directly but has no signed
        # equivalent, and the existing vectors here do the same.
        "logits0": [str(v) for v in logits0[: min(16, a.vocab)]],
        "argmax0": __import__("intref").argmax(logits0),
    }
    path = os.path.join(a.out, "moe_run.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  blob      {hexfile} ({len(raw)} bytes)")
    print(f"  experts   {a.experts}, top-k {a.top_k}")
    print(f"  prompt    {a.prompt}")
    print(f"  generated {tokens}")
    print(f"  argmax0   {out['argmax0']}")
    print(f"  wrote     {path}")


if __name__ == "__main__":
    main()
