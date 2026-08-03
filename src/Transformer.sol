// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {FixedPoint} from "./FixedPoint.sol";
import {WeightStore} from "./WeightStore.sol";

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

    /// @dev Load the token embedding row and scale it up into the Q32 residual stream.
    ///      `rows` is the number of rows in the block holding it — the full vocabulary
    ///      for the whole blob, or 1 for a single-row slice.
    function embed(bytes memory w, Layout memory o, Config memory c, Buffers memory b, uint256 token, uint256 rows)
        internal
        pure
    {
        uint256 shift = _u8(w, o.tokEmb + token);
        uint256 vals = o.tokEmb + rows + token * uint256(c.dim) * 2;
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

    /* A layer splits into three phases at points where the live state is small enough
     * to checkpoint. Only the middle one grows with position, and it divides again by
     * head, which is what keeps any single transaction bounded as context grows.
     *
     *   PREPARE   norm, Q/K/V, RoPE           flat in position
     *   ATTEND    heads [h0, h1)              grows with position, splits by head
     *   PROJECT   W_o, residual, SwiGLU FFN   flat in position
     *
     * Heads never interact: each reads its own query slice and its own KV head, and
     * writes its own slice of attnOut. Splitting them is a reordering of independent
     * work, not an approximation, so it cannot change a single bit of the result. */

    /* ------------------------------ weight slices -----------------------------
     *
     * A phase touches a small, known part of the blob: PREPARE needs one layer's
     * Q/K/V and one position's RoPE row, PROJECT needs one layer's W_o and FFN,
     * and ATTEND needs no weights at all. Materialising all 540KB for any of them
     * costs ~780k gas, most of it quadratic memory expansion, on every transaction.
     *
     * A slice holds the same [shifts][values] structure as the blob, but with only
     * the rows the phase reads. Crucially the phase functions are then called with
     * a Config whose nLayers is 1 and a layer index of 0, which makes every offset
     * they compute land correctly inside the slice — so the sliced and whole-blob
     * paths run the *same code*, and cannot drift apart numerically. */

    /// @dev Copy rows [r0, r0+n) of a tensor block into `dest` at `at`, preserving the
    ///      [shifts][values] layout. Returns bytes written.
    function _sliceRows(
        address[] memory chunks,
        bytes memory dest,
        uint256 at,
        uint256 base,
        uint256 rowsTotal,
        uint256 r0,
        uint256 n,
        uint256 cols
    ) private view returns (uint256) {
        WeightStore.readRangeAt(chunks, base + r0, n, dest, at);
        WeightStore.readRangeAt(chunks, base + rowsTotal + r0 * cols * 2, n * cols * 2, dest, at + n);
        return n + n * cols * 2;
    }

    /// @dev Buffers are padded so a trailing `mload` of a partial row cannot read past
    ///      the allocation; matmul ignores the lanes beyond `cols`.
    uint256 private constant SLICE_PAD = 32;

    /// @notice Weights for one PREPARE: rms_att, W_q/W_k/W_v for layer `l`, RoPE at `pos`
    function slicePrepare(address[] memory chunks, Layout memory o, Config memory c, uint256 l, uint256 pos)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        uint256 dim = c.dim;
        uint256 half = c.hs / 2;
        uint256 n = 1 + dim * 2 // rms_att: one shift, this layer's gains
            + blockSize(dim, dim) // W_q rows
            + blockSize(c.kvd, dim) * 2 // W_k, W_v rows
            + half * 4 * 2; // one RoPE row, cos and sin
        data = new bytes(n + SLICE_PAD);
        uint256 at = 0;

        // rms_att is a single-row block: shift byte, then this layer's slice of gains
        WeightStore.readRangeAt(chunks, o.rmsAtt, 1, data, at);
        WeightStore.readRangeAt(chunks, o.rmsAtt + 1 + l * dim * 2, dim * 2, data, at + 1);
        s.rmsAtt = at;
        at += 1 + dim * 2;

        s.wq = at;
        at += _sliceRows(chunks, data, at, o.wq, uint256(c.nLayers) * dim, l * dim, dim, dim);
        s.wk = at;
        at += _sliceRows(chunks, data, at, o.wk, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim);
        s.wv = at;
        at += _sliceRows(chunks, data, at, o.wv, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim);

        // Only this position's RoPE row, so the caller passes pos = 0
        s.ropeCos = at;
        WeightStore.readRangeAt(chunks, o.ropeCos + pos * half * 4, half * 4, data, at);
        at += half * 4;
        s.ropeSin = at;
        WeightStore.readRangeAt(chunks, o.ropeSin + pos * half * 4, half * 4, data, at);
    }

    /// @notice Weights for one PROJECT: W_o, rms_ffn, and W_1/W_2/W_3 for layer `l`
    function sliceProject(address[] memory chunks, Layout memory o, Config memory c, uint256 l)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        uint256 dim = c.dim;
        uint256 hid = c.hidden;
        uint256 n = blockSize(dim, dim) + 1 + dim * 2 + blockSize(hid, dim) * 2 + blockSize(dim, hid);
        data = new bytes(n + SLICE_PAD);
        uint256 at = 0;

        s.wo = at;
        at += _sliceRows(chunks, data, at, o.wo, uint256(c.nLayers) * dim, l * dim, dim, dim);

        WeightStore.readRangeAt(chunks, o.rmsFfn, 1, data, at);
        WeightStore.readRangeAt(chunks, o.rmsFfn + 1 + l * dim * 2, dim * 2, data, at + 1);
        s.rmsFfn = at;
        at += 1 + dim * 2;

        s.w1 = at;
        at += _sliceRows(chunks, data, at, o.w1, uint256(c.nLayers) * hid, l * hid, hid, dim);
        s.w2 = at;
        at += _sliceRows(chunks, data, at, o.w2, uint256(c.nLayers) * dim, l * dim, dim, hid);
        s.w3 = at;
        at += _sliceRows(chunks, data, at, o.w3, uint256(c.nLayers) * hid, l * hid, hid, dim);
    }

    /// @notice Weights for the classifier: final RMSNorm gains and the whole W_cls
    function sliceClassify(address[] memory chunks, Layout memory o, Config memory c)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        uint256 n = 1 + c.dim * 2 + blockSize(c.vocab, c.dim);
        data = new bytes(n + SLICE_PAD);
        s.rmsFinal = 0;
        WeightStore.readRangeAt(chunks, o.rmsFinal, 1 + c.dim * 2, data, 0);
        s.wcls = 1 + c.dim * 2;
        _sliceRows(chunks, data, s.wcls, o.wcls, c.vocab, 0, c.vocab, c.dim);
    }

    /// @notice The single token-embedding row for `token`
    function sliceEmbed(address[] memory chunks, Layout memory o, Config memory c, uint256 token)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        data = new bytes(1 + c.dim * 2 + SLICE_PAD);
        s.tokEmb = 0;
        _sliceRows(chunks, data, 0, o.tokEmb, c.vocab, token, 1, c.dim);
    }

    /// @notice A Config addressing a one-layer slice: same arithmetic, local offsets
    /// @dev Every field is restated rather than assigning `d = c` and overwriting one.
    ///      Memory structs assign by reference in Solidity, so that shorter form would
    ///      set the *caller's* nLayers to 1 and every later slice would be cut from a
    ///      blob one fifth the real size.
    function sliced(Config memory c) internal pure returns (Config memory d) {
        d = Config({
            dim: c.dim,
            hidden: c.hidden,
            nLayers: 1,
            nHeads: c.nHeads,
            nKv: c.nKv,
            vocab: c.vocab,
            seqLen: c.seqLen,
            hs: c.hs,
            kvd: c.kvd,
            shared: c.shared,
            invSqrtHs: c.invSqrtHs
        });
    }

    /// @notice PREPARE: everything a layer does before attention
    function prepare(bytes memory w, Layout memory o, Config memory c, Buffers memory b, uint256 pos, uint256 l)
        internal
        pure
    {
        uint256 dim = c.dim;
        rmsnorm(w, o.rmsAtt, l * dim, b.resid, b.normed, dim);
        matmul(w, o.wq, uint256(c.nLayers) * dim, l * dim, dim, dim, b.normed, b.query, 0);
        matmul(w, o.wk, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim, b.normed, b.key, 0);
        matmul(w, o.wv, uint256(c.nLayers) * c.kvd, l * c.kvd, c.kvd, dim, b.normed, b.value, 0);
        rope(w, o, c, b.query, c.nHeads, pos);
        rope(w, o, c, b.key, c.nKv, pos);
    }

    /// @notice Write this position's key and value into the in-memory cache
    function cacheKv(Config memory c, Buffers memory b, uint256 l, uint256 pos) internal pure {
        uint256 row = (slotOf(b, l) * b.maxPos + pos) * c.kvd;
        for (uint256 i = 0; i < c.kvd; ++i) {
            b.keyCache[row + i] = b.key[i];
            b.valueCache[row + i] = b.value[i];
        }
    }

    /// @notice ATTEND: heads [h0, h1) for one layer, into their slices of attnOut
    function attend(Config memory c, Buffers memory b, uint256 l, uint256 pos, uint256 h0, uint256 h1) internal pure {
        uint256 kvMul = c.nHeads / c.nKv;
        uint256 base = slotOf(b, l) * b.maxPos * c.kvd;
        for (uint256 h = h0; h < h1; ++h) {
            _head(c, b, base, h * c.hs, (h / kvMul) * c.hs, pos + 1);
        }
    }

    /* ---------------------------- row-sharded PROJECT -------------------------
     *
     * Output rows of a matmul are independent: row i reads all of x and writes one
     * value, touching nothing else. So a matmul splits across transactions exactly,
     * with no partial sums to carry — the only state that crosses a boundary is the
     * vector being filled in.
     *
     * This is what decides whether a model is merely slow or actually impossible.
     * At 1.1B parameters a single FFN matmul is ~11.5M multiply-accumulates, which
     * is two orders of magnitude past any per-transaction budget; sharded by row it
     * is however many transactions it needs to be. */

    /// @notice Multiply-accumulates one sharded matmul phase aims for.
    /// @dev Rows per phase is derived from this and the tensor's width rather than
    ///      fixed, so a wide tensor takes fewer rows per transaction and the cost of
    ///      a phase stays roughly flat no matter the shape. ~150k MACs is ~7.3M gas
    ///      at the measured ~49 gas per MAC, leaving room under a 14M ceiling for the
    ///      slice read and the checkpoint around it.
    uint256 internal constant MACS_PER_PHASE = 150_000;

    /// @notice Rows of a `cols`-wide tensor to process in one phase
    function rowsPerPhase(uint256 macs, uint256 cols, uint256 rowsLeft) internal pure returns (uint256 n) {
        n = macs / cols;
        if (n == 0) n = 1;
        if (n > rowsLeft) n = rowsLeft;
    }

    /// @notice W_o rows [r0, r0+n) of a slice, into proj at the same indices
    function projectWo(bytes memory w, Layout memory s, Config memory c, Buffers memory b, uint256 r0, uint256 n)
        internal
        pure
    {
        matmul(w, s.wo, n, 0, n, c.dim, b.attnOut, b.proj, r0);
    }

    /// @notice W_1 and W_3 rows [r0, r0+n), combined through SwiGLU into gate
    /// @dev Both tensors produce the same output index, and the activation is
    ///      elementwise, so a row range of each is everything this range of `gate`
    ///      needs — `up` never has to exist beyond the phase that creates it.
    function projectFfn1(bytes memory w, Layout memory s, Config memory c, Buffers memory b, uint256 r0, uint256 n)
        internal
        pure
    {
        matmul(w, s.w1, n, 0, n, c.dim, b.normed, b.gate, r0);
        matmul(w, s.w3, n, 0, n, c.dim, b.normed, b.up, r0);
        for (uint256 i = r0; i < r0 + n; ++i) {
            b.gate[i] = (FixedPoint.silu(b.gate[i]) * b.up[i]) >> FixedPoint.F;
        }
    }

    /// @notice W_2 rows [r0, r0+n) of a slice, into proj at the same indices
    function projectFfn2(bytes memory w, Layout memory s, Config memory c, Buffers memory b, uint256 r0, uint256 n)
        internal
        pure
    {
        matmul(w, s.w2, n, 0, n, c.hidden, b.gate, b.proj, r0);
    }

    /// @notice Weights for a W_o shard: rows [r0, r0+n)
    function sliceWo(address[] memory chunks, Layout memory o, Config memory c, uint256 l, uint256 r0, uint256 n)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        data = new bytes(blockSize(n, c.dim) + SLICE_PAD);
        s.wo = 0;
        _sliceRows(chunks, data, 0, o.wo, uint256(c.nLayers) * c.dim, l * c.dim + r0, n, c.dim);
    }

    /// @notice Weights for a SwiGLU shard: W_1 and W_3 rows [r0, r0+n), plus rms_ffn
    function sliceFfn1(address[] memory chunks, Layout memory o, Config memory c, uint256 l, uint256 r0, uint256 n)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        uint256 one = blockSize(n, c.dim);
        data = new bytes(one * 2 + SLICE_PAD);
        s.w1 = 0;
        _sliceRows(chunks, data, 0, o.w1, uint256(c.nLayers) * c.hidden, l * c.hidden + r0, n, c.dim);
        s.w3 = one;
        _sliceRows(chunks, data, one, o.w3, uint256(c.nLayers) * c.hidden, l * c.hidden + r0, n, c.dim);
    }

    /// @notice Weights for a W_2 shard: rows [r0, r0+n)
    function sliceFfn2(address[] memory chunks, Layout memory o, Config memory c, uint256 l, uint256 r0, uint256 n)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        data = new bytes(blockSize(n, c.hidden) + SLICE_PAD);
        s.w2 = 0;
        _sliceRows(chunks, data, 0, o.w2, uint256(c.nLayers) * c.dim, l * c.dim + r0, n, c.hidden);
    }

    /// @notice The rms_ffn gains for one layer, as a single-row block
    function sliceRmsFfn(address[] memory chunks, Layout memory o, Config memory c, uint256 l)
        internal
        view
        returns (bytes memory data, Layout memory s)
    {
        data = new bytes(1 + c.dim * 2 + SLICE_PAD);
        s.rmsFfn = 0;
        WeightStore.readRangeAt(chunks, o.rmsFfn, 1, data, 0);
        WeightStore.readRangeAt(chunks, o.rmsFfn + 1 + l * c.dim * 2, c.dim * 2, data, 1);
    }

    /// @notice PROJECT: output projection, residual, and the feed-forward network
    function project(bytes memory w, Layout memory o, Config memory c, Buffers memory b, uint256 l) internal pure {
        uint256 dim = c.dim;
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

    /// @notice Run layers [lo, hi) for one position, all phases in sequence. This is the
    ///         whole-pass path; the resumable contract calls the phases individually.
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
        if (lo == 0) embed(w, o, c, b, token, c.vocab);
        for (uint256 l = lo; l < hi; ++l) {
            prepare(w, o, c, b, pos, l);
            cacheKv(c, b, l, pos);
            attend(c, b, l, pos, 0, c.nHeads);
            project(w, o, c, b, l);
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
