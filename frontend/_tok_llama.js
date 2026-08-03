/* Llama tokenizer (32k vocabulary), for models that use it — stories15M and up.
 *
 * The merge logic is identical to the tok512 encoder next to it; only the source of
 * the vocabulary differs. It is loaded from llama2.c's tokenizer.bin rather than
 * inlined, because 32,000 tokens are ~178KB of strings and inlining them would
 * roughly double the size of this page for a table most visitors never exercise.
 *
 * The contract stores the same strings on chain and decodes with them, but the table
 * there carries no merge scores — decoding does not need them. Encoding does, so the
 * scores come from this file.
 *
 * Format, little-endian: [i32 maxTokenLength] then per token [f32 score][i32 len][bytes]
 */
function makeLlamaTok(buffer) {
  const dv = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  let off = 4; // maxTokenLength, recomputed from the entries below
  const toks = [];
  const scores = [];
  while (off < buffer.byteLength) {
    scores.push(dv.getFloat32(off, true));
    off += 4;
    const len = dv.getInt32(off, true);
    off += 4;
    toks.push(bytes.subarray(off, off + len));
    off += len;
  }

  const hexOf = (a) => Array.from(a).map((x) => x.toString(16).padStart(2, "0")).join("");
  const keys = toks.map(hexOf);
  // first id wins on duplicates, matching the on-chain table's ordering
  const lookup = new Map();
  keys.forEach((k, i) => { if (!lookup.has(k)) lookup.set(k, i); });

  function encode(text) {
    const raw = new TextEncoder().encode(text);
    const withPrefix = raw.length > 0 ? new Uint8Array([0x20, ...raw]) : raw; // dummy prefix space
    const ids = [1]; // BOS
    for (const c of withPrefix) {
      const id = lookup.get(hexOf([c]));
      ids.push(id === undefined ? c + 3 : id); // byte fallback
    }
    for (;;) {
      let bestScore = -Infinity, bestId = -1, bestIdx = -1;
      for (let i = 1; i + 1 < ids.length; i++) {
        const j = lookup.get(keys[ids[i]] + keys[ids[i + 1]]);
        if (j !== undefined && scores[j] > bestScore) {
          bestScore = scores[j]; bestId = j; bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      ids[bestIdx] = bestId;
      ids.splice(bestIdx + 1, 1);
    }
    return ids;
  }

  function decode(ids, prev) {
    const out = [];
    let p = prev === undefined ? 1 : prev;
    for (const id of ids) {
      let b = toks[id];
      if (p === 1 && b.length && b[0] === 0x20) b = b.subarray(1); // strip space after BOS
      const s = new TextDecoder().decode(b);
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(s);
      out.push(...(m ? [parseInt(m[1], 16)] : Array.from(b)));
      p = id;
    }
    return new TextDecoder().decode(new Uint8Array(out));
  }

  return { encode, decode, size: toks.length };
}
if (typeof module !== "undefined") module.exports = { makeLlamaTok };
