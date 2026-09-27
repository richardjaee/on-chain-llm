"""Mixture-of-experts routing — the specification our Solidity must match.

Extends `intref.py` to the two operations a mixture adds on top of a dense layer:
choosing which experts a token visits, and weighting what they return.

Why a mixture is worth the machinery: gas follows the parameters a token *reads*, and a
dense layer reads all of them. A router that picks 8 experts of 128 skips most of the
layer, and every skipped chunk is a data contract never touched. Storage is billed once;
reading is billed again by every node on every token.

The arithmetic has to be reproducible, as everywhere here. The selection has to be
reproducible too, and that is the sharper requirement: a tie broken differently does not
round differently, it sends the token through different parameters. Two honest nodes then
emit different text and the chain forks. Ties go to the lower index, everywhere, always.
"""

from intref import F, ONE, exp_q32, silu


def top_k(v, k):
    """The `k` largest, as (index, value), highest first. Ties go to the lower index.

    A selection sort rather than a library sort, because the tie rule is the point and a
    comparator that merely looks right is not evidence. Scanning left to right and taking
    a candidate only when it is strictly greater leaves the first occurrence of a repeated
    maximum holding the slot.
    """
    out = []
    k = min(k, len(v))
    for _ in range(k):
        best = None
        for i, x in enumerate(v):
            if any(j == i for j, _ in out):
                continue
            if best is None or x > best[1]:
                best = (i, x)
        if best is not None:
            out.append(best)
    return out


def route_weights(chosen):
    """Softmax over the chosen scores alone, in Q32.

    Over the top `k` rather than over all experts, which is what Qwen3 and DeepSeek do:
    the weights of the experts a token actually visits are renormalised among themselves.
    Normalising over all experts and then dropping the unchosen ones would leave the kept
    weights summing to less than one, quietly scaling the feed-forward output down.
    """
    if not chosen:
        return []
    # Shift by the maximum before exponentiating. Scores are logits and can be large and
    # positive; exp_q32 of a large positive number overflows, and the shift is exact in
    # fixed point, so it costs nothing.
    mx = max(v for _, v in chosen)
    exps = [exp_q32(v - mx) for _, v in chosen]
    total = sum(exps)
    if total == 0:
        # Every exponent underflowed, so the scores are too far apart for the arithmetic
        # to separate. Falling back to the first expert is deterministic; dividing is not.
        return [ONE] + [0] * (len(chosen) - 1)
    return [(e << F) // total for e in exps]


def swiglu(gate, up):
    """SiLU(gate) * up, elementwise, Q32 in and out."""
    return [(silu(g) * u) >> F for g, u in zip(gate, up)]


def visit_order(chosen):
    """Slots of `chosen`, reordered so experts are visited by index.

    The sum over experts is integer and therefore associative, so the order cannot change
    the result. It is fixed anyway: a reader has no way to know the order is irrelevant
    without checking, and one fewer thing to prove about consensus is worth the sort.
    """
    return sorted(range(len(chosen)), key=lambda i: chosen[i][0])
