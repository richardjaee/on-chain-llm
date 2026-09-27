# What it would take to run 1B parameters

A budget, from measured figures rather than estimates. The short version: sparsity is the
whole problem. Every other lever together is worth less than a quarter of what a mixture
is worth on its own.

## Where the numbers come from

| figure | value | source |
|---|---|---|
| gas per multiply-accumulate | 61 | `test_MatmulGas`, three widths |
| gas per parameter per token | 59 at dim 256, 96 at dim 2048 | measured across four widths |
| gas per transaction | 14M | the RPC ceiling, not the block limit |
| today, stories15M | 167 transactions a token | 6 tokens in 1,001 transactions |

The gap between 59 and 96 is per-phase overhead, and it grows with width. At the width a
1B model wants, overhead is already a third of the bill.

## The budget

| | gas a token | transactions a token |
|---|---|---|
| 1B dense, today's kernel | 96B | 6,857 |
| with a mixture at 10% active | 9.6B | 686 |
| and overhead back to 59 gas a parameter | 5.9B | 421 |

421 against today's 167 means a 1B mixture would cost about two and a half times what
stories15M costs now — not two orders of magnitude. That is the whole argument for
believing this is reachable.

## Why the mixture is the only lever that matters

Gas is linear in parameters *read*, not parameters *held*. A mixture token visits a few
experts and never touches the rest, so the bill follows the active count.

This is already measured on the companion project, where the same integer kernel runs
under a modified VM rather than as bytecode:

| model | parameters | gas a token |
|---|---|---|
| Qwen3-0.6B, dense | 0.6B | 277M |
| Qwen3-30B-A3B, mixture | 30B | 870M |

Fifty times the parameters for three times the gas. The integer router there chooses
exactly float64's eight experts at every position, so the selection is not an
approximation that might drift — it is the same selection, computed in integers.

Porting that to Solidity is the project. It needs a router, expert dispatch, and a blob
layout that lets a phase read one expert without paying for the others.

## What does not work, and why

**Wider accumulators packed into one word (SWAR).** The appeal is one `mul` producing
several products in separate lanes. It does not survive the arithmetic. Designing for a
1B model, an `int16` weight times a Q32 activation whose residual peaks near 7,000 is a
2^60 product, and accumulating over dim 2048 reaches 2^71 — so a lane needs 72 bits and a
word holds three. Worse, rows *i* and *i+1* at column *j* are not adjacent in the layout,
so packing them costs an `mload` every two multiply-accumulates where the current layout
spends one every sixteen. Trading a 16:1 amortisation for a 2x saving on the multiply is
a loss.

**Raising the Solidity optimiser's run count.** Measured: three gas out of 20,537,346.
The inner loop is hand-written Yul inside an `assembly` block, which the IR optimiser does
not restructure. There is nothing left there for it to find.

**Transient storage for what a step writes.** Everything a step persists — the residual,
the KV cache, the sharded argmax accumulators, the position and phase counters — has to
survive to the next transaction, and `tstore` does not. No candidate to move.

## The cost that sparsity does not touch

Storage follows total parameters, not active ones.

| | |
|---|---|
| 1B parameters as `int16` | 2.0 GB |
| chunk contracts at 24,576 bytes | 81,380 |
| deployment gas at 200 a byte | 400B |
| upload transactions | 28,571 |

One time, but it is a real 28,571 transactions before the model answers anything, and a
mixture makes it no smaller.

## Order of work

1. **Port the mixture to Solidity.** Worth 10x. Everything else is noise beside it.
2. **Attack per-phase overhead at width**, the 59-to-96 gap. Worth about 1.6x.
3. **Narrow the KV cache to Q16 in `int32`.** Buys context rather than throughput —
   position 286 to roughly 570 — and changes every pinned vector, so it belongs in a
   version that redeploys.
