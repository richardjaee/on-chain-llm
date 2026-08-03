# onchain-llm

A language model that runs **entirely inside Ethereum transactions** — no operators, no
off-chain compute, no trusted anything. The prompt goes in as token ids and every
multiply-accumulate executes as EVM opcodes.

Live on Base Sepolia: [`0xc2ff49E48A5E7e2E472e6858610CF92BFbcD0e1C`](https://sepolia.basescan.org/address/0xc2ff49E48A5E7e2E472e6858610CF92BFbcD0e1C)

```
prompt : Once upon a time          (5 tokens, maxNew = 8)
output : , there was a little girl

52 transactions · 492 phases · ~405M gas · 12 blocks · 25 seconds
```

Measured on Base Sepolia, not simulated. Those figures are for that run, not constants
of the model: cost rises with position as attention reads more of the KV cache, so a
longer generation costs more per transaction from the very first one.

## Why it takes many transactions

A forward pass of this model costs far more gas than any RPC provider accepts in one
transaction — roughly 15M in practice. That limit is separate from the block gas limit,
which on Base Sepolia is 1.2 **billion**. So the model can never be run by simply calling
it, however large blocks get. `eth_call` happily simulates at 1.19B gas, which makes the
wall easy to miss until you send a real transaction.

The fix is to make inference **resumable**, and to make the resumable unit small enough
that nothing can outgrow a transaction. A layer is three phases — projections, attention
one KV head at a time, then the feed-forward network in row-sized pieces — each of which
checkpoints to storage so the next transaction can continue. A transaction runs as many
phases as its gas allows, so the split costs nothing when there is room and rescues the
run when there is not. Total compute is unbounded while every transaction stays small.

The split is exact rather than approximate. Attention heads never interact, matmul output
rows never interact, and what does accumulate does so by integer addition — so dividing
the work reorders independent operations and cannot change a single bit of the result.

## Contracts

| File | Role |
|---|---|
| `OnchainLLM.sol` | Resumable inference: layers per transaction, state checkpointed on-chain |
| `Transformer.sol` | Integer forward pass — RMSNorm, RoPE, GQA attention, SwiGLU, classifier |
| `FixedPoint.sol` | Q32 `isqrt`, `expQ32`, `silu`, `argmax` |
| `WeightStore.sol` | Weights as contract bytecode, with arbitrary-range streaming reads |
| `Decoder.sol` | Token ids back to UTF-8 from an on-chain table |

Weights live as contract bytecode across 22 data contracts (540 KB), read with
`EXTCODECOPY`. Nothing is precomputed — contract storage holds no outputs.

## Design decisions

**Per-row quantisation.** Weights are int16 with a power-of-two scale *per row*. A single
exponent for a whole matrix is set by its loudest row and throws away precision on every
quieter one; measured on this checkpoint, row exponents span 14–18 within one matrix, so
per-tensor would discard up to 4 bits. Costs one byte per row.

**Streaming reads.** EVM memory is quadratic (`3w + w²/512`), so materialising a whole
weight blob passes a 14M budget at about 2.7 MB — capping the model near 1.3M parameters
regardless of how finely the compute is sharded. `readRange` fetches only the rows a shard
needs, making peak memory a constant the caller picks rather than a function of model size.

**Decode-only on-chain.** A greedy BPE encoder costs O(len² · vocab) in gas because the
merge loop rescans the vocabulary for every adjacent pair. Prompts arrive pre-tokenised;
only the cheap decode path is on-chain.

**The classifier is its own step.** Bundling it with the final layer produces the largest
step, and that step grows with position as attention reads more of the KV cache — it would
eventually breach the per-transaction cap and strand a generation mid-flight.

**KV cache packed four values per slot** as int64 lanes, which shrinks both the per-step
write and the per-position read growth.

## Correctness

Splitting a computation across transactions is only legitimate if it changes nothing, so
every layer is pinned to the one below it — and never to another implementation:

- `isqrt` floor-exact over 21 vectors plus a 256-run fuzz
- `expQ32` and `silu` bit-identical to the Python specification
- the on-chain layout derivation spans the 539,963-byte blob **exactly**
- `matmul` with per-row shifts bit-identical to spec
- first-position logits identical, argmax 407
- full generation identical: `[432, 383, 286, 261, 376, 298, 315, 421]`
- the resumable contract reproduces that same sequence across 492 phases
- row-sharded matmuls reproduce it again under a budget that forces every tensor to split

```bash
forge test
```

## Running it

```bash
python3 tools/convert.py --model stories260K.bin      # checkpoint -> artifacts
forge script script/Deploy.s.sol --rpc-url $RPC --private-key $PK --broadcast
```

`frontend/` is a self-contained console: connect a wallet, type a prompt, and watch each
transaction stream in decoded from the contract's own `StepRun` / `TokenEmitted` events, so
you can see which layer ran and which token it produced. It uses a session key, because
Ethereum has no way to authorise a contract to send transactions for you, and batching
calls into one transaction would exceed the very gas cap this design exists to work around.

**Steps are submitted in parallel, not one at a time.** Transactions from a single sender
execute in strict nonce order, which is exactly the ordering `step` requires, so waiting
for each receipt before sending the next buys nothing but latency. The console keeps a
window of transactions in flight and lets the sequencer pack them into as few blocks as
the gas limit allows. This is safe because `step` is self-healing — it advances from
whatever `pos` and `layer` it finds in storage, so a transaction that reverts loses gas
but no work, and the next one performs that step instead.

Measured on Base Sepolia itself, same generation, identical output:

```
serialised   249.2s   48 transactions   124 blocks
batched       21.3s   47 transactions    10 blocks
```

Latency is bounded by blocks, not by transaction count, and a block holds far more than
one step.

There is a catch worth knowing, because it is invisible until you hit it. Base's sequencer
allows **one in-flight transaction for EIP-7702 delegated accounts** — the kind you get
from a smart wallet — rejecting anything further with `in-flight transaction limit reached
for delegated accounts`. Batching from such an account silently degrades to the serialised
row above. The session key sidesteps this because it is a freshly generated EOA with no
delegation, which turns out to be a second reason the design needs one.

## Limits

The gas ceiling is no longer the binding constraint, and neither is transaction count now
that steps are pipelined — a block holds about a hundred of them. What remains is total
compute. At 1B parameters a token needs on the order of 10,000 transactions, which is
~100 blocks of sequencer capacity with nothing else in them, after depositing gigabytes of
weights across ~100,000 contracts.

**Context length** is bounded by whichever phase stops fitting in one transaction. Only
ATTEND grows with position, and it divides by head, so the reachable position is set by
one KV head's share of the cache rather than by a whole layer. Measured at 14M gas per
transaction, the wall moved from position 48 to **187**; `test_FindStrandPosition` pins it.
What remains is the cold read of the KV cache itself, which is linear in position — the
next step is splitting ATTEND over the KV range, in two passes so it stays exact.

Note that the usual answer here — FlashAttention-style online softmax — is the wrong tool.
Its rescaling exists to make attention single-pass, which is unnecessary when state is
already checkpointed between transactions, and the rescaling multiplications would break
bit-exactness against the specification. Heads are independent and the KV range accumulates
by integer addition, so both split *exactly*, with no rescaling at all.

**The split is not free.** Against the same generation the old whole-layer contract used
256M gas in 72 transactions; the phase-split one uses ~405M in 52. Every phase reads its
own weight slice and checkpoints its own state, and a small model never needed the split
to begin with. What it buys is that nothing scales with model size any more: a phase reads
one tensor's rows rather than the whole blob, and a matmul is however many row-sized pieces
it needs to be. At 1.1B parameters a single feed-forward matmul is ~1.9B gas as one unit —
136x any per-transaction ceiling — and the same code now runs it in pieces.

Beware of measuring this the easy way. `test_ResumableMatchesSpec` runs without a gas cap,
so the loop packs every phase into a handful of transactions and the per-transaction
overhead disappears; it reported +6% where the chain reports +58%. Only the capped
benchmark reflects what a real transaction pays.

## Licence and credits

Copyright (c) 2026 Richard Chung. Released under the
[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
licence (`LICENSE`) — free to read, run, modify, and share for any noncommercial purpose,
including personal projects, research, and use by educational or nonprofit institutions.
Commercial use requires permission; open an issue or get in touch.

Note that this is a source-available licence, not an OSI-approved open source one. The
fixed-point formulation, per-row quantisation, weight layout, streaming reader, resumable
execution model, contracts, and frontend are original to this project.

The inference algorithm — RMSNorm, RoPE, grouped-query attention, SwiGLU, greedy decoding —
follows Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT,
`LICENSE-llama2c`). `tools/llama2-reference.c` is that project's `run.c` vendored
unmodified as a readable reference; nothing here compiles against it.

Third-party components keep their own licences, which are unaffected by the terms above
and remain permissive:

| Component | Licence |
|---|---|
| [llama2.c](https://github.com/karpathy/llama2.c) `run.c`, vendored | MIT — `LICENSE-llama2c` |
| [ethers.js](https://github.com/ethers-io/ethers.js) v6.13.4, vendored | MIT — `frontend/LICENSE-ethers` |
| [forge-std](https://github.com/foundry-rs/forge-std), submodule | MIT / Apache-2.0 |

The weights in `artifacts/` are quantised from the `stories260K` checkpoint, distributed
via [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0).
That checkpoint carries no licence file of its own; the licence above covers this
project's code, not the model weights.
