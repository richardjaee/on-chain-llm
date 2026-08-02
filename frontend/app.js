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
const DEFAULT_CONTRACT = "0x77D4fd6B1fcA991d586c75A94008441947B8Eb0b";
const RPCS = ["https://sepolia.base.org", "https://base-sepolia-rpc.publicnode.com"];

/* stories260K geometry — used only for labelling the feed */
const N_LAYERS = 5, N_HEADS = 8, N_KV = 4, DIM = 64, HIDDEN = 172, VOCAB = 512;

const ABI = [
  "function generatedIds() view returns (uint16[])",
  "function start(uint16[] ids, uint256 maxNew)",
  "function step(uint256 reserve)",
  "function text() view returns (string)",
  "function finished() view returns (bool)",
  "function steps() view returns (uint256)",
  "function progress() view returns (uint256 pos, uint256 layer, uint256 maxPos, bool done, uint256 steps)",
  "event StepRun(uint256 indexed pos, uint256 indexed layer, uint256 stepIndex)",
  "event TokenEmitted(uint256 indexed pos, uint16 token, uint256 stepIndex)",
  "event GenerationFinished(uint256 tokens, uint256 totalSteps, string text)",
];
const iface = new E.Interface(ABI);

const STEP_GAS = 14_000_000n, START_GAS = 3_000_000n, RESERVE = 400_000n;
const GAS_PER_STEP = 11_000_000n;

const PROMPT_KEY = "onchain-llm-last-prompt";
/* The session key is persisted (testnet only) so leftover funds survive a page
 * reload. Keeping it purely in memory silently strands whatever is left in the
 * wallet the moment the tab refreshes. */
const SESSKEY = "onchain-llm-session-key";
const lastPrompt = () => { try { return localStorage.getItem(PROMPT_KEY) || ""; } catch (e) { return ""; } };
const rememberPrompt = (t) => { try { localStorage.setItem(PROMPT_KEY, t); } catch (e) {} };

const tok = makeTok512(TOK512_SPEC);
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let provider = null, browserSigner = null, session = null;
let running = false, cancel = false, sessionGas = 0n, feedCount = 0;
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
function status(msg, cls) { $("status").textContent = msg; $("status").className = "status" + (cls ? " " + cls : ""); }

/* ------------------------------- the feed ------------------------------ */
function feedReset() { $("feed").innerHTML = ""; feedCount = 0; clsSeen.clear(); }

/** Describe what a layer step actually computed, in model terms. */
function layerOps(pos) {
  const keys = pos + 1;
  return `RMSNorm → Q/K/V projections (${DIM}→${DIM}/${(DIM / N_HEADS) * N_KV}) → RoPE → ` +
         `causal attention, ${N_HEADS} heads over ${keys} key${keys === 1 ? "" : "s"} → ` +
         `output projection → SwiGLU FFN (${DIM}→${HIDDEN}→${DIM}) → residual`;
}

function addTx({ kind, layer, pos, gas, hash, token, piece, msg, cls }) {
  const el = document.createElement("div");
  el.className = "tx" + (cls ? " " + cls : kind === "classifier" ? " cls" : "");
  feedCount++;
  let head, ops = "", emit = "";
  if (kind === "layer") {
    head = `<span class="tx-kind">Layer ${layer} of ${N_LAYERS}</span>` +
           `<span class="tx-pos">token position ${pos}</span>`;
    ops = `<div class="tx-ops">${layerOps(pos)}</div>`;
  } else if (kind === "classifier") {
    head = `<span class="tx-kind">Classifier</span><span class="tx-pos">token position ${pos}</span>`;
    ops = `<div class="tx-ops">final RMSNorm → logits over ${VOCAB} vocabulary rows → greedy argmax</div>`;
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

/* stepAuto emits the layer index it has advanced TO, not the one it just ran, so
 * `layer = 1` means "finished layer 1 of 5". The value N_LAYERS therefore appears
 * twice per position: once for the final layer, once again for the classifier
 * stage. Ordered processing disambiguates them. */
const clsSeen = new Set();

/** Turn a receipt into feed entries using the contract's own events. */
function describeReceipt(r, hash) {
  let stepRun = null, emitted = null;
  for (const lg of r.logs) {
    let p;
    try { p = iface.parseLog({ topics: [...lg.topics], data: lg.data }); } catch (e) { continue; }
    if (!p) continue;
    if (p.name === "StepRun") stepRun = { pos: Number(p.args.pos), layer: Number(p.args.layer) };
    if (p.name === "TokenEmitted") emitted = { pos: Number(p.args.pos), token: Number(p.args.token) };
  }
  if (!stepRun) { addTx({ kind: "step", msg: "no StepRun event found", gas: r.gasUsed, hash, cls: "note" }); return null; }

  const atEnd = stepRun.layer >= N_LAYERS;
  const key = stepRun.pos;
  const isClassifier = !!emitted || (atEnd && clsSeen.has(key));
  if (atEnd) clsSeen.add(key);

  if (isClassifier) {
    const piece = emitted ? tok.decode([emitted.token], 0) : undefined;
    addTx({ kind: "classifier", pos: stepRun.pos, gas: r.gasUsed, hash,
            token: emitted ? emitted.token : undefined, piece });
    return emitted ? { token: emitted.token, piece } : null;
  }
  // layer value is 1-based already (it is the layer just completed)
  addTx({ kind: "layer", layer: stepRun.layer, pos: stepRun.pos, gas: r.gasUsed, hash });
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
  $("mSteps").textContent = s.steps;
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
async function estimateCost(txCount) {
  const fee = await withFailover(async (p) => (await p.getFeeData()).maxFeePerGas || 2000000n);
  return (BigInt(txCount) * GAS_PER_STEP * fee * 16n) / 10n; // 1.6x for fee drift + L1 data
}
async function ensureFunded(txCount) {
  const need = await estimateCost(txCount);
  const bal = await sessionBalance();
  if (bal >= need) { addTx({ kind: "session already funded", msg: "no approval needed", cls: "note" }); return; }
  const signer = await ensureWallet();
  const top = need - bal;
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
 * Measured against an anvil fork of Base Sepolia at its real 2s block time, one
 * 6-token generation, identical output every time:
 *
 *     serial     219.5s    48 tx   110 blocks
 *     batch 12    18.9s    47 tx    10 blocks
 *     batch 24    10.5s    47 tx     5 blocks
 *     batch 48     6.3s    47 tx     3 blocks
 *     batch 96     8.8s    47 tx     4 blocks   <- submission cost exceeds the saving
 *
 * 32 rather than 48 because nodes cap how many transactions they queue per sender
 * (geth allows 16 executable and 64 queued by default) and anvil does not. Exceeding
 * that is harmless — the fill loop below keeps whatever was accepted and moves on —
 * so this is tuned to sit inside the common limit rather than to probe for it. */
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

    // Fill the batch. Each call is only eth_sendRawTransaction — nothing is mined
    // yet, so these sit in the mempool in nonce order.
    const batch = [];
    for (let i = 0; i < BATCH && !cancel; i++) {
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
        sent++;
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
  const ids = tok.encode(text);
  const maxNew = parseInt($("maxnew").value, 10);
  const txCount = (ids.length + maxNew) * 6;

  running = true; cancel = false; sessionGas = 0n;
  $("go").disabled = true; $("stop").disabled = false;
  feedReset();
  try {
    ownRun = true;
    rememberPrompt(text);
    $("carry").textContent = "";
    $("storyPrompt").textContent = text;
    $("storyOld").textContent = ""; $("storyNew").textContent = "";
    ensureSession();
    addTx({ kind: "tokenized in browser", msg: `${ids.length} ids: [${ids.join(", ")}]`, cls: "note" });
    status("Preparing…", "busy");
    await ensureFunded(txCount);

    status("Starting generation…", "busy");
    const c = new E.Contract(addr(), ABI, session);
    const st = await c.start(ids, maxNew, { gasLimit: START_GAS });
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
  const txs = (ids.length + n) * 6;
  $("maxnewOut").textContent = n;
  $("estimate").textContent =
    `prompt is ${ids.length} tokens · ${n} new ≈ ${txs} transactions, ~${(txs * 11e6 * 0.006e-9).toFixed(4)} ETH ` +
    `(tokens are fragments, so ${n} ≈ ${Math.round(n * 2.5)} characters)`;
}

window.addEventListener("DOMContentLoaded", () => {
  $("addr").value = DEFAULT_CONTRACT;
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
