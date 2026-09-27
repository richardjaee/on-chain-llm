# An OpenAI-shaped endpoint in front of a model that runs on a blockchain

Change a base URL, get a completion, and get a link beside it.

```bash
node serve/index.js
curl -s localhost:8787/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"stories15m-onchain","prompt":"Once upon a time"}'
```

```json
{
  "choices": [{ "index": 0, "text": ", there was a little girl" }],
  "onchain": {
    "verified": true,
    "chain": "base-sepolia",
    "contract": "0x676Ef269b1Fe68129cE7D8001349d6802b9E339e",
    "explorer": "https://sepolia.basescan.org/address/0x676Ef269b1Fe68129cE7D8001349d6802b9E339e",
    "steps": 1276
  }
}
```

## Why the link is the point

A completion arrives the way every other completion arrives. Beside it is the contract
that produced it, and the transactions that did the arithmetic.

Nobody has to install anything to check that. They do not have to re-run a container, or
ask anyone for the weights, or trust that a log was not edited after the fact. They click.

That is the argument for putting a model on a chain rather than merely making it
reproducible. A local receipt says *re-run this and see*; a link says *look*. The second
is the one you can paste into a dispute.

## What `verified` means, exactly

> The arithmetic happened where this says it happened, and anyone re-executing it gets
> identical bits.

It is not a claim that the output is true, safe, or useful. A verified wrong answer is
still wrong. The model here is `stories15M` — Karpathy's TinyStories checkpoint, 15.19M
parameters — so it writes short children's fiction and nothing else.

## Endpoints

| | |
|---|---|
| `GET /v1/models` | the deployed models, with their contracts |
| `POST /v1/completions` | `{prompt, model, max_tokens}` → a job id |
| `GET /v1/completions/:id` | status, then the completion and its proof |

## Why completions are jobs

Six tokens took 1,001 transactions and six and a half minutes. Nothing about that fits a
blocking call that expects a first token inside a second, so a completion returns a job id
and is polled.

This is honest about what is happening rather than a slow server pretending otherwise, and
"your proof is being written to a public chain" earns a wait in a way that nothing else
does. It is also the shape batch and fine-tuning endpoints already use, so a caller has
seen it before.

## Read-only by default

With no `PRIVATE_KEY` the server serves the generation the contract already holds and says
so in the response. That is enough to demonstrate the shape, and it costs nothing.

```bash
PRIVATE_KEY=0x… node serve/index.js   # submits new generations
```

The key needs Base Sepolia gas. Generation is thousands of transactions, so use a
throwaway funded with testnet ether and nothing else.

## Dependencies

None. `ethers` and the tokenizer are the copies the frontend already vendors, loaded
straight from `../frontend`.

## What this is not, yet

Latency and model size are both real. A caller replacing a hosted 70B gets a 15M model
that writes about little girls and takes minutes. The endpoint is the right shape for the
product and the wrong size for the job, and closing that is what
[`docs/scaling-to-1b.md`](../docs/scaling-to-1b.md) is about.
