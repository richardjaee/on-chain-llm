#!/usr/bin/env node
/**
 * An OpenAI-shaped endpoint in front of a model that runs on a blockchain.
 *
 * The point is the last two fields of the response. A completion arrives the way every
 * other completion arrives, and beside it is a link: the contract that produced it, the
 * transactions that did the arithmetic, and a block number. Nobody has to install
 * anything to check it, or ask anyone's permission, or trust that a log was not edited
 * afterwards. They click.
 *
 * That is the whole argument for putting a model on a chain rather than merely making it
 * reproducible. A local receipt says "re-run this and see"; a link says "look".
 *
 * What it does NOT claim: that the output is true, safe, or good. `verified` means the
 * arithmetic happened where it says it happened, and anyone re-executing it gets the same
 * bits. A verified wrong answer is still wrong.
 *
 *   node serve/index.js                 # read-only: serves the completed on-chain run
 *   PRIVATE_KEY=0x… node serve/index.js # drives new generations
 *
 * Generation is slow — six tokens took 1,001 transactions and six and a half minutes —
 * so completions are jobs rather than blocking calls. That is honest about what is
 * happening, and "your proof is being written to a public chain" earns a wait in a way
 * that a slow server does not.
 */

"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

const FRONTEND = path.join(__dirname, "..", "frontend");
const E = require(path.join(FRONTEND, "ethers.umd.min.js"));
const { makeLlamaTok } = require(path.join(FRONTEND, "_tok_llama.js"));

const PORT = Number(process.env.PORT || 8787);

/**
 * Two places the same model can run, and they are not interchangeable.
 *
 * `verified` is a public chain: slow, costs gas, and produces a link anyone can click.
 * `dev` is a local fork of it — the same deployed contract and the same weights, read
 * through anvil — which is free and lets a transaction carry as much work as you allow
 * rather than the ~15M gas a public RPC accepts. Thirty-four phases a transaction
 * instead of one, measured.
 *
 * What `dev` cannot do is prove anything. Those transactions exist on one laptop. So it
 * reports `verified: false` and says why, rather than producing a receipt that looks like
 * the real one. A sandbox that claims what production claims is worse than no sandbox.
 */
const CHAINS = {
  verified: {
    rpc: process.env.RPC || "https://sepolia.base.org",
    explorer: "https://sepolia.basescan.org",
    name: "base-sepolia",
    verified: true,
  },
  dev: {
    rpc: process.env.DEV_RPC || "http://127.0.0.1:8545",
    explorer: null,
    name: "base-sepolia-fork (local)",
    verified: false,
  },
};
const DEFAULT_CHAIN = "verified";

/** Deployed models. The tokenizer is a property of the model, not of this server. */
const MODELS = {
  "stories15m-onchain": {
    address: "0x676Ef269b1Fe68129cE7D8001349d6802b9E339e",
    vocab: 32000,
    tokenizer: "tokenizer-llama.bin",
  },
  "stories260k-onchain": {
    address: "0xc2ff49e48a5E7e2e472e6858610cF92BFbcd0e1C",
    vocab: 512,
    tokenizer: null, // a 512-token table, decoded on chain rather than here
  },
};
const DEFAULT_MODEL = "stories15m-onchain";

const ABI = [
  "function start(uint16[] ids, uint256 maxNew)",
  "function step(uint256 reserve)",
  "function text() view returns (string)",
  "function generatedIds() view returns (uint16[])",
  "function finished() view returns (bool)",
  "function steps() view returns (uint256)",
  "function pos() view returns (uint256)",
  "function maxPos() view returns (uint256)",
];

/** anvil's first account, funded and well known. Only ever used against `dev`. */
const ANVIL_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

function connect(which) {
  const ch = CHAINS[which];
  const provider = new E.JsonRpcProvider(ch.rpc);
  // A local fork is funded by definition, so `dev` needs no key from anyone. The public
  // chain needs a real one, and without it the server reads rather than writes.
  const key = which === "dev" ? process.env.DEV_PRIVATE_KEY || ANVIL_KEY : process.env.PRIVATE_KEY;
  return { ch, provider, wallet: key ? new E.Wallet(key, provider) : null };
}

let tok = null;
function tokenizer() {
  if (!tok) {
    const buf = fs.readFileSync(path.join(FRONTEND, MODELS[DEFAULT_MODEL].tokenizer));
    tok = makeLlamaTok(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  }
  return tok;
}

/** Jobs live in memory: this is a proof of concept, and a restart losing them is fine. */
const jobs = new Map();
let nextId = 1;

function proof(ch, address, extra = {}) {
  return {
    verified: ch.verified,
    claim: ch.verified
      ? "This completion was computed by the contract below, on chain. Anyone can " +
        "re-execute it and get identical bits. It is not a claim that the output is true."
      : "Computed on a local fork. The weights and the contract are the deployed ones, " +
        "but these transactions exist only on this machine, so nothing here is evidence " +
        "to anyone else. Use the verified chain when the receipt is the point.",
    chain: ch.name,
    contract: address,
    explorer: ch.explorer ? `${ch.explorer}/address/${address}` : null,
    ...extra,
  };
}

/** The generation already sitting in the contract — what a read-only server can serve. */
async function readCompleted(model, ch, provider) {
  const c = new E.Contract(model.address, ABI, provider);
  const [text, ids, finished, steps, block] = await Promise.all([
    c.text(),
    c.generatedIds(),
    c.finished(),
    c.steps(),
    provider.getBlockNumber(),
  ]);
  return {
    text,
    tokens: ids.map(Number),
    onchain: proof(ch, model.address, {
      finished,
      steps: Number(steps),
      read_at_block: block,
      note: "Read from contract state; this server did not submit these transactions.",
    }),
  };
}

async function drive(job, model, ids, maxNew, ch, provider, wallet) {
  const c = new E.Contract(model.address, ABI, wallet);
  // A local fork has no meaningful ceiling, so one transaction carries many phases
  // instead of one. This is the whole reason `dev` is worth having.
  const opts = ch.verified ? {} : { gasLimit: 300_000_000n };
  const txs = [];
  try {
    job.status = "starting";
    const st = await c.start(ids, maxNew, opts);
    txs.push(st.hash);
    await st.wait();

    job.status = "generating";
    // The contract advances itself one phase at a time; `step` is called until it says
    // it is done. A reserve of zero lets each transaction use what the node will give it.
    for (let i = 0; i < 20000; ++i) {
      if (await c.finished()) break;
      const t = await c.step(0, opts);
      txs.push(t.hash);
      await t.wait();
      job.steps = txs.length;
      job.pos = Number(await c.pos());
    }

    const [text, gen, blk] = await Promise.all([c.text(), c.generatedIds(), provider.getBlockNumber()]);
    job.status = "done";
    job.result = {
      text,
      tokens: gen.map(Number),
      onchain: proof(ch, model.address, {
        transactions: txs,
        transaction_count: txs.length,
        block: blk,
          first_transaction: txs.length && ch.explorer ? `${ch.explorer}/tx/${txs[0]}` : null,
      }),
    };
  } catch (err) {
    job.status = "error";
    job.error = String(err.shortMessage || err.message || err);
    job.result = { onchain: proof(ch, model.address, { transactions: txs }) };
  }
}

function send(res, code, body) {
  const s = JSON.stringify(body, null, 2);
  res.writeHead(code, { "content-type": "application/json", "access-control-allow-origin": "*" });
  res.end(s);
}

async function body(req) {
  const chunks = [];
  for await (const ch of req) chunks.push(ch);
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString());
  } catch {
    return null;
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (req.method === "GET" && url.pathname === "/v1/models") {
    return send(res, 200, {
      object: "list",
      data: Object.entries(MODELS).map(([id, m]) => ({
        id,
        object: "model",
        owned_by: "on-chain-llm",
        contract: m.address,
        chains: Object.keys(CHAINS),
        explorer: `${CHAINS.verified.explorer}/address/${m.address}`,
      })),
    });
  }

  if (req.method === "POST" && url.pathname === "/v1/completions") {
    const b = await body(req);
    if (!b) return send(res, 400, { error: { message: "body is not JSON" } });

    const name = b.model || DEFAULT_MODEL;
    const model = MODELS[name];
    if (!model) return send(res, 404, { error: { message: `unknown model ${name}` } });

    const which = b.chain || DEFAULT_CHAIN;
    if (!CHAINS[which]) {
      return send(res, 400, { error: { message: `unknown chain ${which}`, chains: Object.keys(CHAINS) } });
    }
    const { ch, provider, wallet } = connect(which);

    // No key means no transactions to sign, so the honest thing is to serve what the
    // contract already holds and say so, rather than pretend to have generated it.
    if (!wallet) {
      const done = await readCompleted(model, ch, provider);
      return send(res, 200, {
        object: "text_completion",
        model: name,
        choices: [{ index: 0, text: done.text, finish_reason: "stop" }],
        onchain: done.onchain,
        warning: "Read-only: set PRIVATE_KEY to submit a new generation.",
      });
    }

    if (!model.tokenizer) {
      return send(res, 400, { error: { message: `${name} needs pre-tokenised ids` } });
    }
    const ids = b.prompt_ids || tokenizer().encode(String(b.prompt ?? ""));
    if (!ids.length) return send(res, 400, { error: { message: "empty prompt" } });

    const id = `cmpl-${nextId++}`;
    const job = { id, model: name, chain: ch.name, status: "queued", created: Date.now(), steps: 0, pos: 0 };
    jobs.set(id, job);
    drive(job, model, ids, Number(b.max_tokens || 6), ch, provider, wallet);

    return send(res, 202, {
      id,
      object: "text_completion.job",
      model: name,
      chain: ch.name,
      verified: ch.verified,
      status: job.status,
      poll: `/v1/completions/${id}`,
      note: ch.verified
        ? "Generation runs as transactions on a public chain and takes minutes, not " +
          "milliseconds. Poll the URL above."
        : "Generation runs against a local fork: free, and one transaction carries many " +
          "phases rather than one. Still minutes, and it proves nothing to anyone else.",
    });
  }

  if (req.method === "GET" && url.pathname.startsWith("/v1/completions/")) {
    const job = jobs.get(url.pathname.split("/").pop());
    if (!job) return send(res, 404, { error: { message: "no such job" } });
    const out = {
      id: job.id,
      object: "text_completion",
      model: job.model,
      chain: job.chain,
      status: job.status,
      progress: { transactions: job.steps, position: job.pos },
    };
    if (job.error) out.error = { message: job.error };
    if (job.result) {
      out.choices = [{ index: 0, text: job.result.text, finish_reason: "stop" }];
      out.onchain = job.result.onchain;
    }
    return send(res, 200, out);
  }

  send(res, 404, { error: { message: "not found" } });
});

server.listen(PORT, () => {
  console.log(`on-chain-llm  ->  http://localhost:${PORT}`);
  console.log(`  model    ${DEFAULT_MODEL} at ${MODELS[DEFAULT_MODEL].address}`);
  for (const [k, c] of Object.entries(CHAINS)) {
    const w = connect(k).wallet;
    console.log(`  ${k.padEnd(8)} ${c.rpc}  ${w ? "signer " + w.address.slice(0, 10) : "read-only"}`);
  }
});
