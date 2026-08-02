"""Integer-only Llama-2 forward pass over per-row-quantised weights.

Structure follows `forward()` in llama2.c (MIT): embed, then per layer
[RMSNorm -> QKV -> RoPE -> causal attention -> W_o -> residual, RMSNorm ->
SwiGLU FFN -> residual], then a final RMSNorm and the classifier.

Everything is integer. This module is the specification the Solidity must
reproduce bit for bit.
"""
from intref import ONE, F, isqrt, exp_q32, silu, argmax, EPS_Q32


class Model:
    def __init__(self, cfg, qw, shifts, cos_t, sin_t):
        self.c, self.w, self.s = cfg, qw, shifts
        self.cos, self.sin = cos_t, sin_t
        c = cfg
        self.inv_sqrt_hs = round((1 << 32) / (c["hs"] ** 0.5))

    def new_state(self, max_pos):
        c, L = self.c, self.c["n_layers"]
        return {"k": [[[0] * c["kvd"] for _ in range(max_pos)] for _ in range(L)],
                "v": [[[0] * c["kvd"] for _ in range(max_pos)] for _ in range(L)]}

    # ---- primitives specialised to per-row shifts ----
    def _rmsnorm(self, x, gname, base, dim):
        g, gs = self.w[gname], self.s[gname][0]
        ss = 0
        for v in x:
            ss += (v * v) >> F
        ss = ss // dim + EPS_Q32
        rms = isqrt(ss << F)
        return [(((x[i] * g[base + i]) << F) // rms) >> gs for i in range(dim)]

    def _matmul(self, name, rowBase, rows, cols, x):
        w, sh = self.w[name], self.s[name]
        out = []
        for i in range(rows):
            r = rowBase + i
            off = r * cols
            acc = 0
            for j in range(cols):
                acc += w[off + j] * x[j]
            out.append(acc >> sh[r])
        return out

    def _rope(self, vec, n_heads, pos):
        hs = self.c["hs"]
        cr, sr = self.cos[pos], self.sin[pos]
        for h in range(n_heads):
            for p in range(hs // 2):
                i = h * hs + 2 * p
                v0, v1 = vec[i], vec[i + 1]
                vec[i] = (v0 * cr[p] - v1 * sr[p]) >> 30
                vec[i + 1] = (v0 * sr[p] + v1 * cr[p]) >> 30

    def forward(self, st, token, pos):
        c = self.c
        dim, hs, kvd = c["dim"], c["hs"], c["kvd"]
        n_heads, n_kv, L = c["n_heads"], c["n_kv"], c["n_layers"]
        kv_mul = n_heads // n_kv

        # token embedding row -> Q32 residual stream
        emb, es = self.w["tok_emb"], self.s["tok_emb"][token]
        x = [v << (F - es) for v in emb[token * dim:(token + 1) * dim]]

        for l in range(L):
            xb = self._rmsnorm(x, "rms_att", l * dim, dim)
            q = self._matmul("wq", l * dim, dim, dim, xb)
            k = self._matmul("wk", l * kvd, kvd, dim, xb)
            v = self._matmul("wv", l * kvd, kvd, dim, xb)
            self._rope(q, n_heads, pos)
            self._rope(k, n_kv, pos)
            st["k"][l][pos], st["v"][l][pos] = k, v

            xatt = [0] * dim
            for h in range(n_heads):
                qo = h * hs
                ko = (h // kv_mul) * hs
                scores = []
                for t in range(pos + 1):
                    kt = st["k"][l][t]
                    acc = 0
                    for j in range(hs):
                        acc += q[qo + j] * kt[ko + j]
                    sc = (acc >> F) * self.inv_sqrt_hs >> 32
                    scores.append(sc)
                mx = max(scores)
                exps = [exp_q32(s - mx) for s in scores]
                tot = sum(exps)
                for j in range(hs):
                    acc = 0
                    for t in range(pos + 1):
                        acc += exps[t] * st["v"][l][t][ko + j]
                    xatt[qo + j] = acc // tot
            xo = self._matmul("wo", l * dim, dim, dim, xatt)
            for i in range(dim):
                x[i] += xo[i]

            xb = self._rmsnorm(x, "rms_ffn", l * dim, dim)
            hid = c["hidden"]
            h1 = self._matmul("w1", l * hid, hid, dim, xb)
            h3 = self._matmul("w3", l * hid, hid, dim, xb)
            hb = [(silu(h1[i]) * h3[i]) >> F for i in range(hid)]
            xo = self._matmul("w2", l * dim, dim, hid, hb)
            for i in range(dim):
                x[i] += xo[i]

        x = self._rmsnorm(x, "rms_final", 0, dim)
        return self._matmul("wcls", 0, c["vocab"], dim, x)

    def generate(self, prompt_ids, n_new):
        max_pos = min(len(prompt_ids) + n_new, self.c["seqlen"])
        st = self.new_state(max_pos)
        out, tok = [], prompt_ids[0]
        for pos in range(max_pos - 1):
            logits = self.forward(st, tok, pos)
            if pos + 1 < len(prompt_ids):
                nxt = prompt_ids[pos + 1]
            else:
                nxt = argmax(logits)
                out.append(nxt)
                if nxt in (1, 2):     # BOS / EOS terminate
                    break
            tok = nxt
        return out
