/* On-chain LLM console — StepwiseLLM on Base Sepolia.
 *
 * Inference is resumable: one transformer layer per transaction, ~6 transactions
 * per token. Confirming each in a wallet is unusable, so a throwaway session key
 * is kept in memory — you approve one funding transfer, everything after is
 * signed locally.
 *
 * Each transaction in the feed is decoded from the contract's own StepRun /
 * TokenEmitted events rather than guessed from call order, so what is displayed
 * is what the chain actually recorded. */

const E = window.ethers;
const CHAIN_ID = 84532;
const CHAIN_ID_HEX = "0x14a34";
const DEFAULT_CONTRACT = "0x676Ef269b1Fe68129cE7D8001349d6802b9E339e";
const RPCS = ["https://sepolia.base.org", "https://base-sepolia-rpc.publicnode.com"];

/* Model geometry is read from the contract, never assumed. It decides how many
 * phases a position costs, which drives both the funding estimate and the batch
 * size — and it is different for every model and every deployment. `phasesPerPosition`
 * is derived on chain from the layer count, KV heads, tensor widths and the
 * deployment's MAC budget, so this console works against a model it has never seen. */
let geom = null;
const PHASE_NAMES = [
  "query projection",
  "key/value projection",
  "attention",
  "output projection",
  "SwiGLU feed-forward",
  "FFN projection",
  "classifier",
];

const ABI = [
  "function generatedIds() view returns (uint16[])",
  "function start(uint16[] ids, uint256 maxNew)",
  "function step(uint256 reserve)",
  "function text() view returns (string)",
  "function finished() view returns (bool)",
  "function steps() view returns (uint256)",
  "function progress() view returns (uint256 pos, uint256 layer, uint256 maxPos, bool done, uint256 steps)",
  "function phasesPerPosition() view returns (uint256)",
  "function config() view returns (tuple(uint32 dim,uint32 hidden,uint32 nLayers,uint32 nHeads,uint32 nKv,uint32 vocab,uint32 seqLen,uint32 hs,uint32 kvd,bool shared,uint64 invSqrtHs))",
  "event StepRun(uint256 indexed pos, uint256 indexed layer, uint256 stepIndex, uint256 phase)",
  "event TokenEmitted(uint256 indexed pos, uint16 token, uint256 stepIndex)",
  "event GenerationFinished(uint256 tokens, uint256 totalSteps, string text)",
];
const iface = new E.Interface(ABI);

const STEP_GAS = 14_000_000n, RESERVE = 400_000n;

/* `start` allocates the residual stream, which is one storage slot per dimension, so
 * its cost scales with the model — it also pre-writes those slots so no later step
 * carries a fresh-slot spike. A fixed 3M was fine for the first model and silently
 * too small for the second. */
const startGas = () => 600_000n + BigInt(geom ? geom.dim : 64) * 7_000n;

/* Rough gas for one phase, from the model's width. Phases are bounded by the MAC
 * budget, but the fixed part of one — slice reads, checkpoints, the norm recompute —
 * grows with dim, and that is what dominates the estimate. Deliberately generous:
 * under-funding strands a run mid-flight, over-funding is recoverable from the
 * session key. Measured ~0.76M at dim 64 and ~5.3M at dim 288. */
const phaseGas = () => 1_200_000n + BigInt(geom ? geom.dim : 64) * 15_000n;

const PROMPT_KEY = "onchain-llm-last-prompt";
/* The session key is persisted (testnet only) so leftover funds survive a page
 * reload. Keeping it purely in memory silently strands whatever is left in the
 * wallet the moment the tab refreshes. */
const SESSKEY = "onchain-llm-session-key";
const lastPrompt = () => { try { return localStorage.getItem(PROMPT_KEY) || ""; } catch (e) { return ""; } };
const rememberPrompt = (t) => { try { localStorage.setItem(PROMPT_KEY, t); } catch (e) {} };

/* Which tokenizer applies is a property of the deployed model, not of this page, so
 * it is chosen once the contract's vocabulary is known. tok512 is the default because
 * it needs no fetch; anything larger loads the Llama vocabulary alongside. */
let tok = makeTok512(TOK512_SPEC);
let tokReady = false;
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let provider = null, browserSigner = null, session = null;
let running = false, cancel = false, sessionGas = 0n, sessionTx = 0, feedCount = 0;
/* The contract keeps whatever the last generation left in storage — possibly from
 * an earlier run or another person entirely. Showing that next to this page's
 * saved prompt implies a relationship that may not exist, so output is only
 * rendered for a run started here. */
let ownRun = false;

function readProvider() {
  if (!provider) provider = new E.JsonRpcProvider(RPCS[0], CHAIN_ID);
  return provider;
}
/* FallbackProvider's 2nd argument is the NETWORK, not a quorum — passing a number
 * there makes every read fail with "network changed". Manual failover instead. */
async function withFailover(fn) {
  let lastErr;
  for (let i = 0; i < RPCS.length; i++) {
    try { return await fn(new E.JsonRpcProvider(RPCS[i], CHAIN_ID)); }
    catch (e) { lastErr = e; }
  }
  throw lastErr;
}
const addr = () => ($("addr").value || DEFAULT_CONTRACT).trim();

/* Switching models invalidates everything derived from the old one: its geometry, and
 * the tokenizer chosen from its vocabulary. Both are re-read on the next Generate. */
function selectModel(a) {
  $("addr").value = a;
  geom = null;
  tokReady = false;
  tok = makeTok512(TOK512_SPEC);
  $("modelNote").textContent = "reading the model from chain…";
  $("nLayersOut").textContent = "—";
  $("phasesOut").textContent = "—";
  $("subModel").textContent = "A transformer";
  ownRun = false;
  updateEstimate();
  loadGeom().then(describeModel).then(refresh).catch((e) => {
    $("modelNote").textContent = "could not read this contract: " + shortErr(e);
  });
}

/** Put the model's own numbers into the page, so nothing here is a hardcoded claim. */
function describeModel() {
  const g = geom;
  const params = g.vocab * g.dim + g.nLayers * (2 * g.dim * g.dim + 2 * g.kvd * g.dim + 3 * g.hidden * g.dim);
  $("subModel").textContent = `A ${(params / 1e6).toFixed(2)}M-parameter transformer`;
  $("nLayersOut").textContent = g.nLayers;
  $("phasesOut").textContent = g.phasesPerPos;
  $("modelNote").textContent =
    `dim ${g.dim} · hidden ${g.hidden} · ${g.nLayers} layers · ${g.nHeads} heads / ${g.nKv} KV · ` +
    `vocab ${g.vocab.toLocaleString()} · ${g.phasesPerPos} phases per token`;
  updateEstimate();
}
function status(msg, cls) { $("status").textContent = msg; $("status").className = "status" + (cls ? " " + cls : ""); }

/* ------------------------------- the feed ------------------------------ */
function feedReset() { $("feed").innerHTML = ""; feedCount = 0; clsSeen.clear(); }

/** Load the geometry once, so nothing about the model is assumed by this page. */
async function loadGeom() {
  if (geom) return geom;
  geom = await withFailover(async (p) => {
    const c = new E.Contract(addr(), ABI, p);
    const [cfg, per] = await Promise.all([c.config(), c.phasesPerPosition()]);
    return {
      dim: Number(cfg.dim), hidden: Number(cfg.hidden), nLayers: Number(cfg.nLayers),
      nHeads: Number(cfg.nHeads), nKv: Number(cfg.nKv), vocab: Number(cfg.vocab),
      seqLen: Number(cfg.seqLen), hs: Number(cfg.hs), kvd: Number(cfg.kvd),
      phasesPerPos: Number(per),
    };
  });
  if (!tokReady) {
    tok = await loadTokenizer(geom.vocab);
    if (tok.size < geom.vocab) {
      throw new Error(`tokenizer has ${tok.size} tokens, the model needs ${geom.vocab}`);
    }
    tokReady = true;
    addTx({ kind: "tokenizer loaded", msg: `${TOKENIZERS[tokKind(geom.vocab)].what}, ${tok.size} tokens`, cls: "note" });
  }
  return geom;
}

/* Which tokenizer a model uses is not derivable from its geometry — two models can
 * share a vocabulary size and disagree about what the ids mean — so it is recorded per
 * deployment, with the vocabulary size as a fallback for an address typed in by hand.
 * Getting it wrong does not fail: the prompt encodes to ids that are valid and mean
 * something else, and the model answers a question nobody asked. */
const TOKENIZERS = {
  tok512: { what: "tok512, sentencepiece", load: async () => makeTok512(TOK512_SPEC) },
  llama: {
    what: "Llama sentencepiece",
    load: async () => makeLlamaTok(await (await fetchAsset("tokenizer-llama.bin")).arrayBuffer()),
  },
  bpe: {
    what: "byte-level BPE",
    load: async () => makeBpeTok(await (await fetchAsset("tokenizer-bpe.json")).json()),
  },
};
const BY_CONTRACT = {
  "0xc2ff49e48a5e7e2e472e6858610cf92bfbcd0e1c": "tok512",
  "0x676ef269b1fe68129ce7d8001349d6802b9e339e": "llama",
};
function tokKind(vocab) {
  const known = BY_CONTRACT[addr().toLowerCase()];
  if (known) return known;
  return vocab <= 512 ? "tok512" : vocab <= 32000 ? "llama" : "bpe";
}
async function fetchAsset(name) {
  const res = await fetch(name);
  if (!res.ok) throw new Error(`${name} not found, and this model needs it`);
  return res;
}
async function loadTokenizer(vocab) {
  return TOKENIZERS[tokKind(vocab)].load();
}

/** Describe what a phase computed, in model terms, from the geometry on chain. */
function phaseOps(phase, pos) {
  const g = geom;
  if (!g) return "";
  const keys = pos + 1;
  switch (phase) {
    case 0:
      return `RMSNorm → W_q rows (${g.dim}→${g.dim}) → RoPE`;
    case 1:
      return `W_k / W_v rows (${g.dim}→${g.kvd}) → RoPE → KV cache write`;
    case 2:
      return `causal attention, ${g.nHeads / g.nKv} head${g.nHeads / g.nKv === 1 ? "" : "s"} ` +
             `over ${keys} key${keys === 1 ? "" : "s"}`;
    case 3:
      return `W_o rows (${g.dim}→${g.dim}) → residual`;
    case 4:
      return `SwiGLU feed-forward, W_1 and W_3 rows (${g.dim}→${g.hidden})`;
    case 5:
      return `W_2 rows (${g.hidden}→${g.dim}) → residual`;
    default:
      return `final RMSNorm → W_cls rows over ${g.vocab} vocabulary entries → running argmax`;
  }
}

function addTx({ kind, layer, pos, gas, hash, token, piece, msg, cls }) {
  const el = document.createElement("div");
  el.className = "tx" + (cls ? " " + cls : kind === "classifier" ? " cls" : "");
  feedCount++;
  let head, ops = "", emit = "";
  if (kind === "layer") {
    head = `<span class="tx-kind">Layer ${layer} of ${geom ? geom.nLayers : "?"}</span>` +
           `<span class="tx-pos">token position ${pos}</span>`;
    ops = `<div class="tx-ops">${msg || ""}</div>`;
  } else if (kind === "classifier") {
    head = `<span class="tx-kind">Classifier</span><span class="tx-pos">token position ${pos}</span>`;
    ops = `<div class="tx-ops">${phaseOps(5, pos)}</div>`;
    if (token !== undefined) {
      emit = `<div class="tx-emit">emitted token <strong>${token}</strong> → <span class="piece">${esc(piece || "")}</span></div>`;
    } else {
      emit = `<div class="tx-emit" style="color:var(--ink-3)">prompt token teacher-forced — classifier skipped</div>`;
    }
  } else {
    head = `<span class="tx-kind">${esc(kind)}</span>`;
    ops = msg ? `<div class="tx-ops">${esc(msg)}</div>` : "";
  }
  el.innerHTML =
    `<div class="tx-head"><span class="tx-n">#${feedCount}</span>${head}` +
    (gas ? `<span class="tx-gas">${Number(gas).toLocaleString()} gas</span>` : "") +
    `</div>${ops}${emit}` +
    (hash ? `<div class="tx-ops"><a href="https://sepolia.basescan.org/tx/${hash}" target="_blank" rel="noopener">${hash.slice(0, 18)}…</a></div>` : "");
  $("feed").prepend(el);
  while ($("feed").children.length > 400) $("feed").lastChild.remove();
}

/* `step` reports the layer and phase it has advanced TO, not the ones it just ran.
 * The classifier is now identified by its phase rather than inferred from a repeated
 * layer index, so no ordering state is needed to tell the two apart. */
const clsSeen = new Set();

/** Turn a receipt into feed entries using the contract's own events. */
function describeReceipt(r, hash) {
  let stepRun = null, emitted = null;
  for (const lg of r.logs) {
    let p;
    try { p = iface.parseLog({ topics: [...lg.topics], data: lg.data }); } catch (e) { continue; }
    if (!p) continue;
    if (p.name === "StepRun") {
      stepRun = { pos: Number(p.args.pos), layer: Number(p.args.layer), phase: Number(p.args.phase) };
    }
    if (p.name === "TokenEmitted") emitted = { pos: Number(p.args.pos), token: Number(p.args.token) };
  }
  if (!stepRun) { addTx({ kind: "step", msg: "no StepRun event found", gas: r.gasUsed, hash, cls: "note" }); return null; }

  const nLayers = geom ? geom.nLayers : Infinity;
  const isClassifier = !!emitted || stepRun.phase === 6 || stepRun.layer >= nLayers;

  if (isClassifier) {
    const piece = emitted ? tok.decode([emitted.token], 0) : undefined;
    addTx({ kind: "classifier", pos: stepRun.pos, gas: r.gasUsed, hash,
            token: emitted ? emitted.token : undefined, piece });
    return emitted ? { token: emitted.token, piece } : null;
  }
  // `layer` and `phase` are what the contract advanced TO, so the label names the
  // work this transaction stopped in front of rather than guessing from call order.
  addTx({
    kind: "layer",
    layer: stepRun.layer,
    pos: stepRun.pos,
    gas: r.gasUsed,
    hash,
    msg: `next: ${PHASE_NAMES[stepRun.phase] || "?"} — ${phaseOps(stepRun.phase, stepRun.pos)}`,
  });
  return null;
}

/* ------------------------------- reading ------------------------------- */
async function readState() {
  return withFailover(async (p) => {
    const c = new E.Contract(addr(), ABI, p);
    const [text, pr, gen] = await Promise.all([c.text(), c.progress(), c.generatedIds()]);
    return { text, pos: Number(pr[0]), layer: Number(pr[1]), maxPos: Number(pr[2]),
             done: pr[3], steps: Number(pr[4]), genCount: gen.length };
  });
}
function paint(s, highlightTail) {
  if (!ownRun) { paintIdle(s); return; }
  const p = lastPrompt();
  $("storyPrompt").textContent = p;
  const t = s.text || "";
  if (highlightTail && t.length) {
    // flash the piece that just landed so you can see tokens arriving
    const cut = Math.max(0, t.length - highlightTail.length);
    $("storyOld").textContent = t.slice(0, cut);
    $("storyNew").textContent = t.slice(cut);
  } else {
    $("storyOld").textContent = t;
    $("storyNew").textContent = "";
  }
  if (!p && !t) $("storyOld").textContent = "…";
  $("mGen").textContent = s.genCount === undefined ? "–" : s.genCount;
  $("mPos").textContent = s.pos + " / " + s.maxPos;
  // The contract counts layers executed, not transactions: `step` runs as many layers
  // as the gas left allows, so steps outruns transactions whenever it fits two in one.
  $("mSteps").textContent = s.steps;
  $("mTx").textContent = sessionTx || "–";
  $("mGas").textContent = sessionGas ? Number(sessionGas).toLocaleString() : "–";
  $("mDone").textContent = s.done ? "complete" : running ? "generating" : s.steps > 0 ? "paused" : "idle";
  $("mDone").className = "val " + (s.done ? "ok" : running ? "busy" : "");
  $("bar").style.width = (s.maxPos ? Math.min(100, (s.pos / s.maxPos) * 100) : 0) + "%";
}
/** Empty state: no generation from this session, so show nothing as ours. */
function paintIdle(s) {
  $("storyPrompt").textContent = "";
  $("storyOld").textContent = "";
  $("storyNew").textContent = "";
  $("mGen").textContent = "–";
  $("mPos").textContent = "–";
  $("mSteps").textContent = "–";
  $("mTx").textContent = "–";
  $("mGas").textContent = "–";
  $("mDone").textContent = "idle";
  $("mDone").className = "val";
  $("bar").style.width = "0%";
  const held = s && s.text ? s.text : "";
  $("carry").textContent = held
    ? `The contract currently holds ${held.length} characters from an earlier run. Press Generate to start your own.`
    : "No generation yet. Press Generate to start one.";
}

async function refresh() {
  try {
    const s = await readState();
    if (ownRun) paint(s); else paintIdle(s);
  } catch (e) {
    paintIdle(null);
    addTx({ kind: "read failed", msg: e.shortMessage || e.message, cls: "err" });
  }
}

/** Discard this session's run so the view returns to a clean slate. */
function clearView() {
  ownRun = false;
  sessionGas = 0n;
  feedReset();
  refresh();
}

/* ------------------------- wallet + session key ------------------------ */
async function ensureWallet() {
  if (browserSigner) return browserSigner;
  if (!window.ethereum) throw new Error("No wallet found — install MetaMask, then reload.");
  const bp = new E.BrowserProvider(window.ethereum);
  await bp.send("eth_requestAccounts", []);
  const net = await bp.getNetwork();
  if (Number(net.chainId) !== CHAIN_ID) {
    try { await bp.send("wallet_switchEthereumChain", [{ chainId: CHAIN_ID_HEX }]); }
    catch (e) {
      const code = e.code || (e.error && e.error.code);
      if (code === 4902) {
        await bp.send("wallet_addEthereumChain", [{
          chainId: CHAIN_ID_HEX, chainName: "Base Sepolia",
          nativeCurrency: { name: "Ether", symbol: "ETH", decimals: 18 },
          rpcUrls: ["https://sepolia.base.org"],
          blockExplorerUrls: ["https://sepolia.basescan.org"],
        }]);
      } else throw e;
    }
  }
  browserSigner = await new E.BrowserProvider(window.ethereum).getSigner();
  const a = await browserSigner.getAddress();
  $("acct").textContent = a.slice(0, 6) + "…" + a.slice(-4);
  return browserSigner;
}
function ensureSession() {
  if (session) return session;
  let pk = null;
  try { pk = localStorage.getItem(SESSKEY); } catch (e) {}
  try {
    session = pk ? new E.Wallet(pk, readProvider()) : E.Wallet.createRandom().connect(readProvider());
  } catch (e) {
    session = E.Wallet.createRandom().connect(readProvider());
  }
  try { localStorage.setItem(SESSKEY, session.privateKey); } catch (e) {}
  $("sessAddr").textContent = session.address;
  $("sessKey").textContent = session.privateKey;
  sessionBalance();
  return session;
}

function newSessionKey() {
  try { localStorage.removeItem(SESSKEY); } catch (e) {}
  session = null;
  ensureSession();
  addTx({ kind: "new session key", msg: session.address, cls: "note" });
}
async function sessionBalance() {
  if (!session) return 0n;
  const b = await withFailover((p) => p.getBalance(session.address));
  $("sessBal").textContent = (+E.formatEther(b)).toFixed(5) + " ETH";
  return b;
}
/* Total cost of `positions` positions, from the phase count the contract reports and
 * a per-phase estimate derived from its width. The old form multiplied a transaction
 * count by a fixed 11M, which stopped meaning anything once a transaction packed
 * several phases and a phase stopped costing 11M. */
async function estimateCost(positions) {
  const fee = await withFailover(async (p) => (await p.getFeeData()).maxFeePerGas || 2000000n);
  const gas = BigInt(positions) * BigInt(geom.phasesPerPos) * phaseGas();
  return (gas * fee * 13n) / 10n; // 1.3x for fee drift and L1 data
}
async function ensureFunded(positions) {
  const need = await estimateCost(positions);
  const bal = await sessionBalance();
  if (bal >= need) { addTx({ kind: "session already funded", msg: "no approval needed", cls: "note" }); return; }
  const signer = await ensureWallet();
  const from = await signer.getAddress();
  const have = await withFailover((p) => p.getBalance(from));
  // Never ask for more than the wallet holds. The estimate is deliberately generous
  // and a large model can exceed a testnet balance outright; asking anyway produces a
  // failed gas estimation whose error names nothing useful.
  const keep = 2_000_000_000_000_000n; // leave 0.002 for the wallet's own gas
  let top = need - bal;
  if (have < top + keep) {
    top = have > keep ? have - keep : 0n;
    if (top === 0n) {
      throw new Error(
        `wallet holds ${(+E.formatEther(have)).toFixed(4)} ETH, this run needs about ` +
        `${(+E.formatEther(need)).toFixed(4)} ETH — reduce the token count`
      );
    }
    addTx({
      kind: "partial funding",
      msg: `run needs about ${(+E.formatEther(need)).toFixed(4)} ETH, funding ${(+E.formatEther(top)).toFixed(4)} ` +
           `— it will stop when the session runs out and can be resumed`,
      cls: "err",
    });
  }
  status(`Approve once to fund the run (${(+E.formatEther(top)).toFixed(4)} ETH)…`, "busy");
  const t = await signer.sendTransaction({ to: session.address, value: top });
  await t.wait();
  addTx({ kind: "session funded", msg: `${(+E.formatEther(top)).toFixed(5)} ETH — the only approval`, hash: t.hash, cls: "note" });
  await sessionBalance();
}

/* --------------------------------- run --------------------------------- */

/* Transactions from one sender execute in strict nonce order, and that ordering is
 * exactly what `step` needs — so awaiting each receipt before submitting the next buys
 * nothing but latency. Submitting a batch and then collecting receipts lets the
 * sequencer pack many steps into a single block: a step peaks near 11.9M gas against
 * Base Sepolia's 1.2B block limit, so what bounds a batch is mempool policy, not the
 * chain. Serialised, a run costs one block of latency per transaction; batched, it
 * costs roughly one block per batch.
 *
 * This is only safe because `step` is self-healing. It reads pos and layer from
 * storage and advances from wherever it finds them, so a transaction that reverts
 * loses its gas but no work — the next one in the batch performs that step instead.
 * Running past the end reverts with AlreadyFinished before touching anything.
 *
 * Measured on Base Sepolia itself, one 6-token generation, identical output:
 *
 *     serialised   249.2s    48 tx   124 blocks
 *     batch 32      21.3s    47 tx    10 blocks
 *
 * Base accepts 32 queued transactions from a session key without complaint. It will
 * not do so from an EIP-7702 delegated account — a smart wallet — which is capped at
 * one in flight and rejects the rest with "in-flight transaction limit reached for
 * delegated accounts". Batching from such an account degrades to the serialised row
 * above and nothing else breaks, but it is why the steps must come from the session
 * key rather than from the user's own wallet.
 *
 * Providers differ in how deep a queue they hold, so exceeding the limit has to be
 * survivable: the fill loop keeps whatever was accepted and moves on. */
const BATCH = 32;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shortErr = (err) => ((err && (err.shortMessage || err.message)) || String(err)).slice(0, 140);

/* Optimistic local state between authoritative reads, so tokens still appear one at a
 * time while a batch drains. Every batch ends by replacing this with the contract's
 * own view, so drift lasts at most one batch. */
function advance(s, info) {
  const n = { ...s, steps: s.steps + 1 };
  if (info && info.piece) {
    n.text = (n.text || "") + info.piece;
    n.genCount = (n.genCount || 0) + 1;
    n.pos = n.pos + 1;
  }
  return n;
}

async function runSteps() {
  const c = new E.Contract(addr(), ABI, session);
  let nonce = await withFailover((p) => p.getTransactionCount(session.address, "pending"));
  let sent = 0, stalls = 0;
  let live = await readState();

  for (;;) {
    if (cancel) { addTx({ kind: "stopped", msg: "progress is saved on-chain", cls: "note" }); break; }
    if (live.done) break;
    if (stalls >= 4) { addTx({ kind: "giving up", msg: "no progress across several batches", cls: "err" }); break; }
    const before = live;

    // Price the batch once. Left to itself ethers fetches fee data per transaction,
    // which is a round-trip per step — the very cost this batching exists to remove.
    const fee = await withFailover((p) => p.getFeeData());
    const price = fee.maxFeePerGas
      ? { maxFeePerGas: fee.maxFeePerGas, maxPriorityFeePerGas: fee.maxPriorityFeePerGas ?? 0n }
      : {};

    // Never queue more steps than the run can still use. Overshoot is harmless — the
    // extra transactions revert with AlreadyFinished before touching state — but each
    // one still pays intrinsic gas, and a full batch of them is pure waste.
    const left = Math.max(1, (live.maxPos - live.pos) * geom.phasesPerPos);
    const want = Math.min(BATCH, left);

    // Fill the batch. Each call is only eth_sendRawTransaction — nothing is mined
    // yet, so these sit in the mempool in nonce order.
    const batch = [];
    for (let i = 0; i < want && !cancel; i++) {
      try {
        batch.push(await c.step(RESERVE, { gasLimit: STEP_GAS, nonce, ...price }));
        nonce++;
      } catch (err) {
        // Providers cap how many queued transactions they hold per sender. Work with
        // whatever was accepted; only treat it as a stall if nothing got through.
        if (batch.length === 0) {
          stalls++;
          addTx({ kind: "submission rejected", msg: shortErr(err), cls: "err" });
          nonce = await withFailover((p) => p.getTransactionCount(session.address, "pending"));
          await sleep(2000);
        }
        break;
      }
    }
    if (batch.length === 0) continue;
    status(`${batch.length} transactions in flight…`, "busy");

    // Collect in submission order, so the feed reads the way the chain executed.
    for (const t of batch) {
      try {
        const r = await t.wait();
        sent++; sessionTx++;
        sessionGas += r.gasUsed;
        const info = describeReceipt(r, t.hash);
        live = advance(live, info);
        paint(live, info && info.piece ? info.piece : null);
        status(`Generating… position ${live.pos}/${live.maxPos} · ${live.genCount} tokens · ${sent} transactions`, "busy");
      } catch (err) {
        // A missing receipt is not a failed step. The authoritative read below is the
        // only trustworthy account of what actually landed.
        addTx({ kind: "no receipt", msg: shortErr(err), cls: "err" });
      }
    }

    live = await readState();
    paint(live);
    stalls = live.steps > before.steps ? 0 : stalls + 1;
    // Resync rather than trust the local counter: a dropped submission leaves a gap.
    nonce = await withFailover((p) => p.getTransactionCount(session.address, "pending"));
  }
  return sent;
}

async function generate() {
  if (running) return;
  const text = $("prompt").value.trim();
  if (!text) { status("Type a prompt first.", "bad"); return; }

  running = true; cancel = false; sessionGas = 0n; sessionTx = 0;
  $("go").disabled = true; $("stop").disabled = false;
  feedReset();
  try {
    ownRun = true;
    // Reading the geometry and choosing a tokenizer belong inside the handler: they
    // touch the network and the contract address, so they can fail, and a failure
    // outside it leaves the button looking dead with nothing written anywhere.
    status("Reading model geometry…", "busy");
    await loadGeom();
    const ids = tok.encode(text);
    const maxNew = parseInt($("maxnew").value, 10);
    const positions = ids.length + maxNew;
    rememberPrompt(text);
    updateEstimate();
    $("carry").textContent = "";
    $("storyPrompt").textContent = text;
    $("storyOld").textContent = ""; $("storyNew").textContent = "";
    ensureSession();
    addTx({ kind: "tokenized in browser", msg: `${ids.length} ids: [${ids.join(", ")}]`, cls: "note" });
    status("Preparing…", "busy");
    await ensureFunded(positions);

    status("Starting generation…", "busy");
    const c = new E.Contract(addr(), ABI, session);
    const st = await c.start(ids, maxNew, { gasLimit: startGas() });
    const sr = await st.wait();
    sessionGas += sr.gasUsed;
    addTx({ kind: "generation started", msg: `prompt ${ids.length} tokens, generating up to ${maxNew}`, gas: sr.gasUsed, hash: st.hash, cls: "note" });

    const sent = await runSteps();
    const final = await readState();
    paint(final);
    if (final.done) status(`Done — ${final.genCount} tokens in ${sent} transactions, ${Number(sessionGas).toLocaleString()} gas`, "ok");
    else status(`Paused at position ${final.pos}/${final.maxPos} — press Generate to resume`, "warn");
  } catch (e) {
    const m = e.shortMessage || e.message || String(e);
    status("Stopped: " + m, "bad");
    addTx({ kind: "error", msg: m, cls: "err" });
  } finally {
    running = false;
    $("go").disabled = false; $("stop").disabled = true;
    await sessionBalance();
  }
}

async function sweep() {
  if (!session) return;
  const signer = await ensureWallet();
  const to = await signer.getAddress();
  const bal = await withFailover((p) => p.getBalance(session.address));
  const fee = await withFailover(async (p) => (await p.getFeeData()).maxFeePerGas || 2000000n);
  const cost = fee * 21000n * 3n;
  if (bal <= cost) { addTx({ kind: "nothing to return", cls: "note" }); return; }
  const t = await session.sendTransaction({ to, value: bal - cost });
  await t.wait();
  addTx({ kind: "funds returned", msg: `${(+E.formatEther(bal - cost)).toFixed(5)} ETH`, hash: t.hash, cls: "note" });
  await sessionBalance();
}

function updateEstimate() {
  const n = parseInt($("maxnew").value, 10);
  const ids = tok.encode($("prompt").value.trim() || " ");
  $("maxnewOut").textContent = n;
  if (!geom) {
    $("estimate").textContent = `prompt is ${ids.length} tokens · ${n} new — press Generate to read the model's cost`;
    return;
  }
  const positions = ids.length + n;
  const gas = BigInt(positions) * BigInt(geom.phasesPerPos) * phaseGas();
  $("estimate").textContent =
    `prompt is ${ids.length} tokens · ${n} new ≈ ${positions * geom.phasesPerPos} phases, ` +
    `~${(Number(gas) * 0.006e-9).toFixed(4)} ETH (tokens are fragments, so ${n} ≈ ${Math.round(n * 2.5)} characters)`;
}

window.addEventListener("DOMContentLoaded", () => {
  $("addr").value = DEFAULT_CONTRACT;
  $("model").value = DEFAULT_CONTRACT;
  $("model").onchange = (e) => selectModel(e.target.value);
  selectModel(DEFAULT_CONTRACT);
  $("go").onclick = generate;
  $("stop").onclick = () => { cancel = true; $("stop").disabled = true; status("Stopping after this step…", "warn"); };
  $("refresh").onclick = refresh;
  $("clear").onclick = clearView;
  $("sweep").onclick = () => sweep().catch((e) => addTx({ kind: "return failed", msg: e.shortMessage || e.message, cls: "err" }));
  $("newkey").onclick = newSessionKey;
  // restore any previous session wallet so leftover funds stay recoverable
  try { if (localStorage.getItem(SESSKEY)) ensureSession(); } catch (e) {}
  $("maxnew").oninput = updateEstimate;
  $("prompt").oninput = updateEstimate;
  updateEstimate();
  refresh();
  setInterval(() => { if (!running) refresh(); }, 15000);
});
