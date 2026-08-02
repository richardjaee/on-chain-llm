// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {FixedPoint} from "./FixedPoint.sol";

/// @title Transformer
/// @notice Integer-only Llama-2 forward pass: RMSNorm, RoPE, grouped-query attention,
///         SwiGLU feed-forward, and a greedy classifier — every operation in Q32
///         fixed point over int16 weights with per-row power-of-two scales.
/// @dev Structure follows `forward()` in Karpathy's llama2.c (MIT). The fixed-point
///      formulation and the per-row quantisation are original to this project; the
///      Python specification this must reproduce bit for bit is tools/model.py.
///
///      Blob layout, all big-endian, offsets derived from the config alone:
///        per tensor, in order tok_emb, rms_att, wq, wk, wv, wo, rms_ffn,
///        w1, w2, w3, rms_final, [wcls if unshared]:
///            [u8 shift per row][int16 values, row-major]
///        then RoPE: [int32 cos, seqlen x hs/2][int32 sin, seqlen x hs/2]
library Transformer {
    using FixedPoint for int256;

    struct Config {
        uint32 dim;
        uint32 hidden;
        uint32 nLayers;
        uint32 nHeads;
        uint32 nKv;
        uint32 vocab;
        uint32 seqLen;
        uint32 hs;
        uint32 kvd;
        bool shared;
        uint64 invSqrtHs; // 2^32 / sqrt(hs)
    }

    /// @notice Byte offsets of each tensor's shift table within the blob
    struct Layout {
        uint256 tokEmb;
        uint256 rmsAtt;
        uint256 wq;
        uint256 wk;
        uint256 wv;
        uint256 wo;
        uint256 rmsFfn;
        uint256 w1;
        uint256 w2;
        uint256 w3;
        uint256 rmsFinal;
        uint256 wcls;
        uint256 ropeCos;
        uint256 ropeSin;
    }

    /// @dev Size of one tensor block: `rows` shift bytes plus rows*cols int16
    function blockSize(uint256 rows, uint256 cols) internal pure returns (uint256) {
        return rows + rows * cols * 2;
    }

    /// @notice Derive every offset from the config; the layout is fully determined
    function layout(Config memory c) internal pure returns (Layout memory o) {
        uint256 L = c.nLayers;
        uint256 at = 0;
        o.tokEmb = at;
        at += blockSize(c.vocab, c.dim);
        o.rmsAtt = at;
        at += blockSize(1, L * c.dim);
        o.wq = at;
        at += blockSize(L * c.dim, c.dim);
        o.wk = at;
        at += blockSize(L * c.kvd, c.dim);
        o.wv = at;
        at += blockSize(L * c.kvd, c.dim);
        o.wo = at;
        at += blockSize(L * c.dim, c.dim);
        o.rmsFfn = at;
        at += blockSize(1, L * c.dim);
        o.w1 = at;
        at += blockSize(L * c.hidden, c.dim);
        o.w2 = at;
        at += blockSize(L * c.dim, c.hidden);
        o.w3 = at;
        at += blockSize(L * c.hidden, c.dim);
        o.rmsFinal = at;
        at += blockSize(1, c.dim);
        // A shared classifier reuses the embedding block; an unshared one gets its own
        o.wcls = c.shared ? o.tokEmb : at;
        if (!c.shared) at += blockSize(c.vocab, c.dim);
        o.ropeCos = at;
        at += uint256(c.seqLen) * (c.hs / 2) * 4;
        o.ropeSin = at;
    }

    /// @dev int16 at byte `off` of the blob, sign-extended
    function _i16(bytes memory w, uint256 off) internal pure returns (int256 v) {
        assembly ("memory-safe") {
            v := signextend(1, shr(240, mload(add(add(w, 0x20), off))))
        }
    }

    /// @dev int32 at byte `off`, sign-extended (RoPE tables)
    function _i32(bytes memory w, uint256 off) internal pure returns (int256 v) {
        assembly ("memory-safe") {
            v := signextend(3, shr(224, mload(add(add(w, 0x20), off))))
        }
    }

    /// @dev u8 at byte `off` (per-row shifts)
    function _u8(bytes memory w, uint256 off) internal pure returns (uint256 v) {
        assembly ("memory-safe") {
            v := byte(0, mload(add(add(w, 0x20), off)))
        }
    }

    /// @notice out[i] = sum_j W[row0+i][j] * x[j], requantised by that row's shift
    /// @param base Byte offset of the tensor block (its shift table)
    /// @param rowsTotal Total rows in the tensor, needed to find the value region
    function matmul(
        bytes memory w,
        uint256 base,
        uint256 rowsTotal,
        uint256 row0,
        uint256 rows,
        uint256 cols,
        int256[] memory x,
        int256[] memory out,
        uint256 outOff
    ) internal pure {
        // One EVM word holds sixteen int16 weights, so the inner loop loads a word
        // once and unpacks all sixteen rather than issuing an mload per weight. That
        // removes fifteen of every sixteen memory reads from the hottest loop in the
        // model; the tail handles a column count that is not a multiple of sixteen.
        assembly ("memory-safe") {
            let wp := add(w, 0x20)
            let shiftBase := add(wp, base)
            let valBase := add(shiftBase, rowsTotal) // values follow the shift table
            let xp := add(x, 0x20)
            let op := add(add(out, 0x20), shl(5, outOff))
            let rowBytes := shl(1, cols)
            let whole := and(cols, not(15)) // columns covered by full words

            for { let i := 0 } lt(i, rows) { i := add(i, 1) } {
                let r := add(row0, i)
                let sh := byte(0, mload(add(shiftBase, r)))
                let rp := add(valBase, mul(r, rowBytes))
                let acc := 0

                for { let j := 0 } lt(j, whole) { j := add(j, 16) } {
                    let word := mload(add(rp, shl(1, j)))
                    let xj := add(xp, shl(5, j))
                    // unrolled: constant shifts, no inner loop counter
                    acc := add(acc, mul(signextend(1, shr(240, word)), mload(add(xj, 0))))
                    acc := add(acc, mul(signextend(1, shr(224, word)), mload(add(xj, 32))))
                    acc := add(acc, mul(signextend(1, shr(208, word)), mload(add(xj, 64))))
                    acc := add(acc, mul(signextend(1, shr(192, word)), mload(add(xj, 96))))
                    acc := add(acc, mul(signextend(1, shr(176, word)), mload(add(xj, 128))))
                    acc := add(acc, mul(signextend(1, shr(160, word)), mload(add(xj, 160))))
                    acc := add(acc, mul(signextend(1, shr(144, word)), mload(add(xj, 192))))
                    acc := add(acc, mul(signextend(1, shr(128, word)), mload(add(xj, 224))))
                    acc := add(acc, mul(signextend(1, shr(112, word)), mload(add(xj, 256))))
                    acc := add(acc, mul(signextend(1, shr(96, word)), mload(add(xj, 288))))
                    acc := add(acc, mul(signextend(1, shr(80, word)), mload(add(xj, 320))))
                    acc := add(acc, mul(signextend(1, shr(64, word)), mload(add(xj, 352))))
                    acc := add(acc, mul(signextend(1, shr(48, word)), mload(add(xj, 384))))
                    acc := add(acc, mul(signextend(1, shr(32, word)), mload(add(xj, 416))))
                    acc := add(acc, mul(signextend(1, shr(16, word)), mload(add(xj, 448))))
                    acc := add(acc, mul(signextend(1, shr(0, word)), mload(add(xj, 480))))
                }
                // tail: fewer than sixteen columns remain, so walk the final word's
                // leading lanes with the same shift arithmetic as above
                if lt(whole, cols) {
                    let tailWord := mload(add(rp, shl(1, whole)))
                    let xt := add(xp, shl(5, whole))
                    for { let k := 0 } lt(k, sub(cols, whole)) { k := add(k, 1) } {
                        let lane := signextend(1, shr(sub(240, shl(4, k)), tailWord))
                        acc := add(acc, mul(lane, mload(add(xt, shl(5, k)))))
                    }
                }
                mstore(add(op, shl(5, i)), sar(sh, acc))
            }
        }
    }

    /// @notice x_i * g_i / sqrt(mean(x^2) + eps), all Q32
    /// @dev `gBase` is the 1-row tensor block; `gOff` indexes within its values.
    function rmsnorm(bytes memory w, uint256 gBase, uint256 gOff, int256[] memory x, int256[] memory out, uint256 dim)
        internal
        pure
    {
        uint256 gShift = _u8(w, gBase);
        uint256 vals = gBase + 1;
        int256 ss = 0;
        for (uint256 i = 0; i < dim; ++i) {
            ss += (x[i] * x[i]) >> FixedPoint.F;
        }
        ss = ss / int256(dim) + FixedPoint.EPS;
        int256 rms = int256(FixedPoint.isqrt(uint256(ss) << FixedPoint.F));
        for (uint256 i = 0; i < dim; ++i) {
            int256 g = _i16(w, vals + (gOff + i) * 2);
            out[i] = ((x[i] * g) << FixedPoint.F) / rms >> gShift;
        }
    }

    /// @notice Rotate consecutive pairs within each head using the Q30 RoPE tables
    function rope(bytes memory w, Layout memory o, Config memory c, int256[] memory v, uint256 nHeads, uint256 pos)
        internal
        pure
    {
        uint256 half = c.hs / 2;
        uint256 rowOff = pos * half * 4;
        for (uint256 h = 0; h < nHeads; ++h) {
            for (uint256 p = 0; p < half; ++p) {
                int256 cs = _i32(w, o.ropeCos + rowOff + p * 4);
                int256 sn = _i32(w, o.ropeSin + rowOff + p * 4);
                uint256 i = h * c.hs + 2 * p;
                int256 v0 = v[i];
                int256 v1 = v[i + 1];
                v[i] = (v0 * cs - v1 * sn) >> 30;
                v[i + 1] = (v0 * sn + v1 * cs) >> 30;
            }
        }
    }

    /// @notice Per-run activation buffers and the KV cache
    struct Buffers {
        int256[] resid; // residual stream carried through the layers
        int256[] normed; // RMSNorm output feeding the projections
        int256[] query;
        int256[] key;
        int256[] value;
        int256[] attnOut; // concatenated head outputs
        int256[] proj; // scratch for W_o and W_2 results
        int256[] gate; // SwiGLU gate branch, reused for the activated product
        int256[] up; // SwiGLU up branch
        int256[] logits;
        int256[] keyCache;
        int256[] valueCache;
        int256[] attnWeights;
        uint256 maxPos;
        /// @dev Number of layers the KV cache holds. The resumable path runs one layer
        ///      per call and reloads from storage each time, so it allocates a single
        ///      layer and indexes it at slot 0; a whole-pass caller needs all of them.
        uint256 cacheLayers;
    }

    /// @dev Which slot of the KV cache layer `l` occupies
    function slotOf(Buffers memory b, uint256 l) internal pure returns (uint256) {
        return b.cacheLayers == 1 ? 0 : l;
    }

    /// @notice Buffers holding every layer's KV cache, for running a whole pass at once
    function newBuffers(Config memory c, uint256 maxPos) internal pure returns (Buffers memory b) {
        return newBuffers(c, maxPos, c.nLayers);
    }

    /// @notice Buffers holding `cacheLayers` layers of KV cache over `maxPos` positions.
    /// @dev The KV cache dominates this allocation and EVM memory is quadratic, so
    ///      sizing it to what the call actually touches — rather than to the worst case
    ///      the session might eventually reach — is worth real gas on every transaction.
    function newBuffers(Config memory c, uint256 maxPos, uint256 cacheLayers)
        internal
        pure
        returns (Buffers memory b)
    {
        b.resid = new int256[](c.dim);
        b.normed = new int256[](c.dim);
        b.query = new int256[](c.dim);
        b.key = new int256[](c.kvd);
        b.value = new int256[](c.kvd);
        b.attnOut = new int256[](c.dim);
        b.proj = new int256[](c.dim);
        b.gate = new int256[](c.hidden);
        b.up = new int256[](c.hidden);
        b.logits = new int256[](c.vocab);
        b.keyCache = new int256[](cacheLayers * maxPos * c.kvd);
        b.valueCache = new int256[](cacheLayers * maxPos * c.kvd);
        b.attnWeights = new int256[](maxPos);
        b.maxPos = maxPos;
        b.cacheLayers = cacheLayers;
    }

    /// @dev Load the token embedding row and scale it up into the Q32 residual stream
    function embed(bytes memory w, Layout memory o, Config memory c, Buffers memory b, uint256 token) internal pure {
        uint256 shift = _u8(w, o.tokEmb + token);
        uint256 vals = o.tokEmb + c.vocab + token * uint256(c.dim) * 2;
        uint256 up = FixedPoint.F - shift;
        for (uint256 i = 0; i < c.dim; ++i) {
            b.resid[i] = _i16(w, vals + i * 2) << up;
        }
    }

    /// @dev One attention head: scores, softmax, then the value-weighted sum
    function _head(Config memory c, Buffers memory b, uint256 layerBase, uint256 qOff, uint256 kvOff, uint256 steps)
        private
        pure
    {
        int256 mx = type(int256).min;
        for (uint256 t = 0; t < steps; ++t) {
            uint256 kb = layerBase + t * c.kvd + kvOff;
            int256 acc = 0;
            for (uint256 j = 0; j < c.hs; ++j) {
                acc += b.query[qOff + j] * b.keyCache[kb + j];
            }
            int256 sc = (acc >> FixedPoint.F) * int256(uint256(c.invSqrtHs)) >> FixedPoint.F;
            b.attnWeights[t] = sc;
            if (sc > mx) mx = sc;
        }
        uint256 total = 0;
        for (uint256 t = 0; t < steps; ++t) {
            uint256 e = FixedPoint.expQ32(b.attnWeights[t] - mx);
            b.attnWeights[t] = int256(e);
            total += e;
        }
        for (uint256 j = 0; j < c.hs; ++j) {
            int256 acc = 0;
            for (uint256 t = 0; t < steps; ++t) {
                acc += b.attnWeights[t] * b.valueCache[layerBase + t * c.kvd + kvOff + j];
            }
            b.attnOut[qOff + j] = acc / int256(total);
        }
    }

    /// @notice Run layers [lo, hi) for one position. Splitting is safe because the
    ///         residual stream and KV cache fully describe the state between layers.
    function forwardRange(
        bytes memory w,
        Layout memory o,
        Config memory c,
        Buffers memory b,
        uint256 token,
        uint256 pos,
        uint256 lo,
        uint256 hi
    ) internal pure {
        uint256 dim = c.dim;
        if (lo == 0) embed(w, o, c, b, token);

        for (uint256 l = lo; l < hi; ++l) {
            rmsnorm(w, o.rmsAtt, l * dim, b.resid, b.normed, dim);
            matmul(w, o.wq, uint256(c.nLayers) * dim, l * dim, dim, dim, b.normed, b.query, 0);
            matmul(w, o.wk, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim, b.normed, b.key, 0);
            matmul(w, o.wv, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim, b.normed, b.value, 0);
            rope(w, o, c, b.query, c.nHeads, pos);
            rope(w, o, c, b.key, c.nKv, pos);

            uint256 row = (slotOf(b, l) * b.maxPos + pos) * c.kvd;
            for (uint256 i = 0; i < c.kvd; ++i) {
                b.keyCache[row + i] = b.key[i];
                b.valueCache[row + i] = b.value[i];
            }

            uint256 kvMul = c.nHeads / c.nKv;
            for (uint256 h = 0; h < c.nHeads; ++h) {
                _head(c, b, slotOf(b, l) * b.maxPos * c.kvd, h * c.hs, (h / kvMul) * c.hs, pos + 1);
            }
            matmul(w, o.wo, uint256(c.nLayers) * dim, l * dim, dim, dim, b.attnOut, b.proj, 0);
            for (uint256 i = 0; i < dim; ++i) {
                b.resid[i] += b.proj[i];
            }

            rmsnorm(w, o.rmsFfn, l * dim, b.resid, b.normed, dim);
            matmul(w, o.w1, uint256(c.nLayers) * c.hidden, l * c.hidden, c.hidden, dim, b.normed, b.gate, 0);
            matmul(w, o.w3, uint256(c.nLayers) * c.hidden, l * c.hidden, c.hidden, dim, b.normed, b.up, 0);
            for (uint256 i = 0; i < c.hidden; ++i) {
                b.gate[i] = (FixedPoint.silu(b.gate[i]) * b.up[i]) >> FixedPoint.F;
            }
            matmul(w, o.w2, uint256(c.nLayers) * dim, l * dim, dim, c.hidden, b.gate, b.proj, 0);
            for (uint256 i = 0; i < dim; ++i) {
                b.resid[i] += b.proj[i];
            }
        }
    }

    /// @notice Final RMSNorm, leaving the normed vector in b.normed
    function normFinal(bytes memory w, Layout memory o, Config memory c, Buffers memory b) internal pure {
        rmsnorm(w, o.rmsFinal, 0, b.resid, b.normed, c.dim);
    }

    /// @notice Classifier rows [lo, hi) into b.logits
    function classifyRange(bytes memory w, Layout memory o, Config memory c, Buffers memory b, uint256 lo, uint256 hi)
        internal
        pure
    {
        matmul(w, o.wcls, c.vocab, lo, hi - lo, c.dim, b.normed, b.logits, lo);
    }
}
