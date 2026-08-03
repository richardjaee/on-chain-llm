// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Decoder} from "./Decoder.sol";
import {FixedPoint} from "./FixedPoint.sol";
import {Transformer} from "./Transformer.sol";
import {WeightStore} from "./WeightStore.sol";

/// @title OnchainLLM
/// @notice A transformer that runs across MANY transactions instead of one.
///
/// A single forward pass costs far more gas than any RPC provider will accept in one
/// transaction (~15M in practice), a limit entirely separate from the block gas limit.
/// So the model can never be run by simply calling it, no matter how large blocks are.
/// This contract removes that ceiling by making inference resumable: each `step` runs
/// one or more transformer layers for the current position and checkpoints the residual
/// stream plus that layer's KV-cache row into storage. The next transaction reloads and
/// continues. Total compute is unbounded; each transaction stays small.
///
/// Nothing is computed off-chain. The caller supplies pre-tokenised prompt ids — a
/// greedy BPE encoder costs O(len^2 * vocab) in gas, while decoding is a table lookup —
/// and every multiply-accumulate happens in the EVM.
contract OnchainLLM {
    /// @notice Directory data contract: token table pointer followed by weight chunks
    address public immutable directory;

    uint32 private immutable _dim;
    uint32 private immutable _hidden;
    uint32 private immutable _nLayers;
    uint32 private immutable _nHeads;
    uint32 private immutable _nKv;
    uint32 private immutable _vocab;
    uint32 private immutable _seqLen;
    bool private immutable _shared;

    // --------------------------------------------------------------- session
    uint16[] public promptIds;
    uint16[] public generated;
    uint256 public pos;
    uint256 public layer;
    uint256 public maxPos;
    bool public finished;
    uint256 public steps;

    /// @notice Which phase of the current layer runs next: 0 PREPARE, 1 ATTEND, 2 PROJECT
    uint256 public phase;
    /// @notice Next attention head to run, while `phase` is ATTEND
    uint256 public head;

    /// @dev Residual stream carried between layer steps (Q32)
    int256[] private _x;

    /// @dev KV cache, four Q32 values packed per slot as int64 lanes.
    ///      Key: (isValue ? 1<<128 : 0) + ((layer * seqLen + pos) * slotsPerRow) + lane.
    ///      Packing matters twice: it shrinks the per-step write, and it shrinks the
    ///      per-position READ growth that would otherwise push long sequences past the
    ///      per-transaction gas cap.
    mapping(uint256 => uint256) private _kv;
    uint256 private constant PACK_N = 4;

    /// @dev State that must survive between phases of one layer: the post-RoPE query,
    ///      and the attention output as heads fill it in. Both are Q32 activations, so
    ///      they pack four to a slot as int64 lanes exactly like the KV cache.
    mapping(uint256 => uint256) private _scratch;
    uint256 private constant SCRATCH_AO = 1 << 128;

    /// @dev An ATTEND phase covers the query heads that share one KV head (nHeads/nKv
    ///      under grouped-query attention). Splitting finer would make two phases read
    ///      the same KV slice, and that read is the part that grows with position.

    event GenerationStarted(uint256 promptLen, uint256 maxPos);
    event StepRun(uint256 indexed pos, uint256 indexed layer, uint256 stepIndex);
    event TokenEmitted(uint256 indexed pos, uint16 token, uint256 stepIndex);
    event GenerationFinished(uint256 tokens, uint256 totalSteps);

    error NotStarted();
    error AlreadyFinished();
    error PromptTooLong();
    error KvOverflow(int256 value);

    constructor(address directory_, Transformer.Config memory c) {
        directory = directory_;
        _dim = c.dim;
        _hidden = c.hidden;
        _nLayers = c.nLayers;
        _nHeads = c.nHeads;
        _nKv = c.nKv;
        _vocab = c.vocab;
        _seqLen = c.seqLen;
        _shared = c.shared;
    }

    function config() public view returns (Transformer.Config memory c) {
        c.dim = _dim;
        c.hidden = _hidden;
        c.nLayers = _nLayers;
        c.nHeads = _nHeads;
        c.nKv = _nKv;
        c.vocab = _vocab;
        c.seqLen = _seqLen;
        c.hs = _dim / _nHeads;
        c.kvd = c.hs * _nKv;
        c.shared = _shared;
        // 2^32 / sqrt(hs), via the same integer sqrt the model uses
        c.invSqrtHs = uint64((uint256(1) << 64) / FixedPoint.isqrt(uint256(c.hs) << 64));
    }

    /// @notice Begin a generation from a pre-tokenised prompt
    function start(uint16[] calldata ids, uint256 maxNew) external {
        Transformer.Config memory c = config();
        if (ids.length == 0 || ids.length >= c.seqLen) revert PromptTooLong();
        delete promptIds;
        for (uint256 i = 0; i < ids.length; ++i) {
            promptIds.push(ids[i]);
        }
        delete generated;
        uint256 mp = ids.length + maxNew;
        if (mp > c.seqLen) mp = c.seqLen;
        maxPos = mp;
        pos = 0;
        layer = 0;
        phase = 0;
        head = 0;
        finished = false;
        steps = 0;
        _x = new int256[](c.dim);
        emit GenerationStarted(ids.length, mp);
    }

    /// @notice Advance as far as the remaining gas allows, then checkpoint.
    /// @dev Stage `nLayers` is the classifier, run as its own step: bundling it with
    ///      the last layer produces the largest step, and that step grows with position
    ///      as attention reads more of the KV cache, so it would eventually breach the
    ///      per-transaction cap and strand a generation.
    /// @param reserve Gas to keep back for writing state before returning
    function step(uint256 reserve) external {
        if (maxPos == 0) revert NotStarted();
        if (finished) revert AlreadyFinished();

        Transformer.Config memory c = config();
        Transformer.Layout memory o = Transformer.layout(c);
        (address tokPtr, address[] memory chunks) = _dir();
        tokPtr;
        bytes memory w = _weights(chunks, o, c);

        uint256 p = pos;
        uint256 lo = layer;
        // Size the KV cache to this transaction, not to the session: one layer, because
        // every layer is loaded from and stored back to storage individually, and only
        // the positions attention will actually read. EVM memory is quadratic, so the
        // difference between this and `nLayers * maxPos` is worth hundreds of thousands
        // of gas on every transaction — and the saving grows with the requested maxNew.
        Transformer.Buffers memory b = Transformer.newBuffers(c, lo == c.nLayers ? 1 : p + 1, 1);
        int256[] memory xs = _x;
        for (uint256 i = 0; i < c.dim; ++i) {
            b.resid[i] = xs[i];
        }

        if (lo == c.nLayers) {
            // classifier stage — skipped entirely while still inside the prompt,
            // where the next token is teacher-forced and the logits are discarded
            steps += 1;
            if (p + 1 >= promptIds.length) {
                Transformer.normFinal(w, o, c, b);
                Transformer.classifyRange(w, o, c, b, 0, c.vocab);
            }
            emit StepRun(p, lo, steps);
            _emit(b, p, p + 1 < promptIds.length);
            return;
        }

        if (lo == 0 && phase == 0 && head == 0) {
            uint256 tok0 = p < promptIds.length ? promptIds[p] : generated[p - promptIds.length];
            Transformer.embed(w, o, c, b, tok0);
        }

        // Guard on the most expensive phase seen so far, not the last one. Phases
        // differ by an order of magnitude — an ATTEND costs a fraction of a PROJECT —
        // so the previous phase's cost predicts nothing about the next, and estimating
        // from it walks straight into a PROJECT with no gas left. The high-water mark
        // packs slightly less per transaction and never strands a layer mid-flight.
        //
        // The measured cost also UNDER-estimates the next transaction: within one tx
        // the KV slots are warm (100 gas) where a fresh tx re-reads them cold (2100),
        // hence the additional margin.
        uint256 worst = 0;
        uint256 ran = 0;
        bool residDirty = false;
        while (layer < c.nLayers) {
            uint256 before = gasleft();
            if (ran > 0 && before < worst + worst / 3 + reserve) break;
            if (_phase(w, o, c, b, p)) residDirty = true;
            uint256 cost = before - gasleft();
            if (cost > worst) worst = cost;
            unchecked {
                ran += 1;
            }
        }

        // PREPARE and ATTEND leave the residual stream untouched, so it is only worth
        // writing back when a PROJECT actually ran.
        if (residDirty) {
            for (uint256 i = 0; i < c.dim; ++i) {
                xs[i] = b.resid[i];
            }
            _x = xs;
        }
        steps += ran;
        emit StepRun(p, layer, steps);
    }

    /// @dev Run one phase of the current layer and advance the cursors.
    ///      Returns true if the residual stream changed.
    function _phase(
        bytes memory w,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p
    ) private returns (bool residChanged) {
        uint256 l = layer;
        uint256 ph = phase;

        if (ph == 0) {
            Transformer.prepare(w, o, c, b, p, l);
            _storeKvRow(c, b, l, p);
            _saveVec(0, b.query, 0, c.dim);
            phase = 1;
            head = 0;
            return false;
        }

        if (ph == 1) {
            uint256 h0 = head;
            uint256 h1 = h0 + c.nHeads / c.nKv;
            if (h1 > c.nHeads) h1 = c.nHeads;
            _loadVec(0, b.query, 0, c.dim);
            // Only this head range's slice of the KV cache is touched, so only that
            // slice is read back — the saving grows with position and is what keeps
            // an ATTEND phase affordable deep into a sequence.
            _loadKvHeads(c, b, l, p, h0, h1);
            Transformer.attend(c, b, l, p, h0, h1);
            _saveVec(SCRATCH_AO, b.attnOut, h0 * c.hs, (h1 - h0) * c.hs);
            head = h1;
            if (h1 == c.nHeads) phase = 2;
            return false;
        }

        _loadVec(SCRATCH_AO, b.attnOut, 0, c.dim);
        Transformer.project(w, o, c, b, l);
        phase = 0;
        head = 0;
        layer = l + 1;
        return true;
    }

    /// @dev Save `len` values of `src` from `off`, packed four int64 lanes per slot.
    ///      `off` and `len` are multiples of PACK_N for every caller here (head slices
    ///      are `hs` wide, and `hs` is a multiple of four).
    function _saveVec(uint256 space, int256[] memory src, uint256 off, uint256 len) private {
        for (uint256 i = 0; i < len; i += PACK_N) {
            _scratch[space + (off + i) / PACK_N] = _pack(src, off + i, PACK_N);
        }
    }

    function _loadVec(uint256 space, int256[] memory dst, uint256 off, uint256 len) private view {
        for (uint256 i = 0; i < len; i += PACK_N) {
            uint256 packed = _scratch[space + (off + i) / PACK_N];
            for (uint256 j = 0; j < PACK_N; ++j) {
                dst[off + i + j] = _lane(packed, j);
            }
        }
    }

    /// @dev Pick the next token (or teacher-force from the prompt) and advance
    function _emit(Transformer.Buffers memory b, uint256 p, bool inPrompt) private {
        uint256 next;
        bool stop = false;
        if (inPrompt) {
            next = promptIds[p + 1];
        } else {
            next = FixedPoint.argmax(b.logits);
            generated.push(uint16(next));
            emit TokenEmitted(p, uint16(next), steps);
            if (next == Decoder.BOS || next == Decoder.EOS) stop = true;
        }
        layer = 0;
        pos = p + 1;
        if (stop || pos + 1 > maxPos - 1) {
            finished = true;
            emit GenerationFinished(generated.length, steps);
        }
    }

    /// @notice Decode the generated tokens on-chain
    function text() public view returns (string memory) {
        if (generated.length == 0) return "";
        (address tokPtr,) = _dir();
        bytes memory table = WeightStore.read(tokPtr);
        uint16[] memory ids = generated;
        return string(Decoder.decode(table, ids, ids.length, promptIds[promptIds.length - 1]));
    }

    function generatedIds() external view returns (uint16[] memory) {
        return generated;
    }

    function progress()
        external
        view
        returns (uint256 pos_, uint256 layer_, uint256 maxPos_, bool done_, uint256 steps_)
    {
        return (pos, layer, maxPos, finished, steps);
    }

    // ------------------------------------------------------------- internals
    function _dir() private view returns (address tokPtr, address[] memory chunks) {
        bytes memory d = WeightStore.read(directory);
        uint256 n = d.length / 20 - 1;
        tokPtr = _addr(d, 0);
        chunks = new address[](n);
        for (uint256 i = 0; i < n; ++i) {
            chunks[i] = _addr(d, (i + 1) * 20);
        }
    }

    function _addr(bytes memory d, uint256 off) private pure returns (address a) {
        assembly ("memory-safe") {
            a := shr(96, mload(add(add(d, 0x20), off)))
        }
    }

    function _weights(address[] memory chunks, Transformer.Layout memory o, Transformer.Config memory c)
        private
        view
        returns (bytes memory w)
    {
        uint256 len = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;
        w = new bytes(len);
        WeightStore.readRange(chunks, 0, len, w);
    }

    function _slots(Transformer.Config memory c) private pure returns (uint256) {
        return (c.kvd + PACK_N - 1) / PACK_N;
    }

    function _key(Transformer.Config memory c, bool isV, uint256 l, uint256 p, uint256 lane)
        private
        pure
        returns (uint256)
    {
        uint256 base = ((l * uint256(c.seqLen)) + p) * _slots(c) + lane;
        return isV ? base + (1 << 128) : base;
    }

    function _pack(int256[] memory src, uint256 off, uint256 n) private pure returns (uint256 packed) {
        for (uint256 j = 0; j < n; ++j) {
            int256 v = src[off + j];
            if (v < type(int64).min || v > type(int64).max) revert KvOverflow(v);
            packed |= uint256(uint64(int64(v))) << (64 * j);
        }
    }

    function _lane(uint256 packed, uint256 j) private pure returns (int256) {
        return int256(int64(uint64(packed >> (64 * j))));
    }

    /// @dev Load the KV rows that heads [h0, h1) will read, for every position up to and
    ///      including `p`. Grouped-query attention means head h reads kv head h/kvMul, so
    ///      a head range maps to a contiguous slice of each row rather than the whole of
    ///      it. Reading only that slice is what stops the per-transaction cost of ATTEND
    ///      from growing with the full kv width as the sequence lengthens.
    function _loadKvHeads(
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 l,
        uint256 p,
        uint256 h0,
        uint256 h1
    ) private view {
        uint256 kvMul = c.nHeads / c.nKv;
        uint256 lo = (h0 / kvMul) * c.hs;
        uint256 hi = ((h1 - 1) / kvMul + 1) * c.hs;
        uint256 kvd = c.kvd;
        for (uint256 t = 0; t <= p; ++t) {
            uint256 row = (Transformer.slotOf(b, l) * b.maxPos + t) * kvd;
            for (uint256 i = lo; i < hi; i += PACK_N) {
                uint256 ks = _kv[_key(c, false, l, t, i / PACK_N)];
                uint256 vs = _kv[_key(c, true, l, t, i / PACK_N)];
                for (uint256 j = 0; j < PACK_N; ++j) {
                    b.keyCache[row + i + j] = _lane(ks, j);
                    b.valueCache[row + i + j] = _lane(vs, j);
                }
            }
        }
    }

    function _storeKvRow(Transformer.Config memory c, Transformer.Buffers memory b, uint256 l, uint256 p) private {
        uint256 kvd = c.kvd;
        for (uint256 i = 0; i < kvd; i += PACK_N) {
            uint256 n = kvd - i < PACK_N ? kvd - i : PACK_N;
            _kv[_key(c, false, l, p, i / PACK_N)] = _pack(b.key, i, n);
            _kv[_key(c, true, l, p, i / PACK_N)] = _pack(b.value, i, n);
        }
    }
}
