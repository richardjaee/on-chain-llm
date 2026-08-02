# onchain-llm

A language model that runs **entirely inside Ethereum transactions** — no operators, no
off-chain compute, no trusted anything. The prompt goes in as token ids and every
multiply-accumulate executes as EVM opcodes.

Live on Base Sepolia: [`0x77D4fd6B1fcA991d586c75A94008441947B8Eb0b`](https://sepolia.basescan.org/address/0x77D4fd6B1fcA991d586c75A94008441947B8Eb0b)

```
prompt : Once upon a time
output : , there was a little girl

72 transactions · 256,410,750 gas · peak 11,928,695 per transaction
```

## Why it takes many transactions

A forward pass of this model costs far more gas than any RPC provider accepts in one
transaction — roughly 15M in practice. That limit is separate from the block gas limit,
which on Base Sepolia is 1.2 **billion**. So the model can never be run by simply calling
it, however large blocks get. `eth_call` happily simulates at 1.19B gas, which makes the
wall easy to miss until you send a real transaction.

The fix is to make inference **resumable**. Each transaction runs one or more transformer
layers and checkpoints the residual stream plus that layer's KV-cache row into contract
storage; the next transaction reloads and continues. Total compute is unbounded while
every individual transaction stays small.

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
- the resumable contract reproduces that same sequence across 72 transactions

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

## Limits

The gas ceiling is no longer the binding constraint; cost and latency are. At 1B parameters
a token needs on the order of 10,000 transactions and minutes of block time, after
depositing gigabytes of weights across ~100,000 contracts. Attention is not yet sharded
over the KV range (online softmax), which bounds context length at large model dimensions.

## Licence and credits

MIT. The model is Andrej Karpathy's `stories260K` checkpoint trained on TinyStories, and
the inference algorithm — RMSNorm, RoPE, grouped-query attention, SwiGLU, greedy decoding —
follows [llama2.c](https://github.com/karpathy/llama2.c) (MIT, `LICENSE-llama2c`). The
fixed-point formulation, per-row quantisation, weight layout, streaming reader, and the
resumable execution model are original to this project.
