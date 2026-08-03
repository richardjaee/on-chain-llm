/* Byte-level BPE, the GPT-2 lineage — SmolLM2 and most HuggingFace models.
 *
 * Two things separate it from the sentencepiece tokenizers beside it. Text is split by
 * a regex before merging, so a merge can never span a word boundary. And every byte is
 * mapped to a printable character first, so a leading space is stored as U+0120 and the
 * vocabulary contains no raw bytes at all — which is why the on-chain table stores the
 * bytes a token stands for rather than these characters, leaving Decoder.sol unaware
 * that any of this exists.
 *
 * Merges are ranked by position, not scored: the earliest pair in the list wins, where
 * sentencepiece takes the highest score. Using the wrong rule produces a valid but
 * different tokenisation, which the model would answer as if it were a different prompt.
 */
function makeBpeTok(spec) {
  const toks = spec.tokens;
  const rank = new Map();
  spec.merges.forEach((m, i) => rank.set(m, i));
  const ids = new Map();
  toks.forEach((t, i) => { if (!ids.has(t)) ids.set(t, i); });

  // the GPT-2 byte <-> printable-character table
  const bs = [];
  for (let i = 33; i < 127; i++) bs.push(i);
  for (let i = 161; i < 173; i++) bs.push(i);
  for (let i = 174; i < 256; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const b2u = new Map(), u2b = new Map();
  bs.forEach((b, i) => { b2u.set(b, String.fromCharCode(cs[i])); u2b.set(String.fromCharCode(cs[i]), b); });

  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+/gu;

  function encode(text) {
    const out = [];
    for (const piece of text.match(PAT) || []) {
      let word = Array.from(new TextEncoder().encode(piece)).map((b) => b2u.get(b));
      while (word.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i + 1 < word.length; i++) {
          const r = rank.get(word[i] + " " + word[i + 1]);
          if (r !== undefined && r < best) { best = r; at = i; }
        }
        if (at < 0) break;
        word.splice(at, 2, word[at] + word[at + 1]);
      }
      for (const w of word) {
        const id = ids.get(w);
        if (id !== undefined) out.push(id);
      }
    }
    return out;
  }

  function decode(idList) {
    const chars = idList.map((i) => toks[i] || "").join("");
    const bytes = [];
    for (const ch of chars) {
      const b = u2b.get(ch);
      if (b !== undefined) bytes.push(b);
      else for (const x of new TextEncoder().encode(ch)) bytes.push(x); // control tokens
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  return { encode, decode, size: toks.length };
}
if (typeof module !== "undefined") module.exports = { makeBpeTok };
