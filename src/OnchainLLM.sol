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

    /// @notice Multiply-accumulates a sharded matmul phase aims for. Lower means more,
    ///         smaller transactions — the right value depends on the chain's per
    ///         transaction gas ceiling, so it is set at deployment rather than compiled
    ///         in. Zero takes the library default.
    uint256 public immutable macsPerPhase;

    // --------------------------------------------------------------- session
    uint16[] public promptIds;
    uint16[] public generated;
    uint256 public pos;
    uint256 public layer;
    uint256 public maxPos;
    bool public finished;
    uint256 public steps;

    /// @notice Which phase of the current layer runs next.
    ///         0 W_q, 1 W_k/W_v, 2 ATTEND, 3 W_o, 4 SwiGLU, 5 W_2
    uint256 public phase;
    /// @notice Position within the current phase: the next head for ATTEND, the next
    ///         output row for the matmul phases, the next vocabulary row for the
    ///         classifier.
    uint256 public cursor;

    /// @dev Running argmax over the vocabulary, carried across classifier shards. The
    ///      logits themselves are never stored: each shard scans its own rows and keeps
    ///      only whichever is best so far, which is what makes a 32k vocabulary a
    ///      bounded amount of state rather than 32k words of it. `>` rather than `>=`
    ///      keeps the first maximum winning, matching FixedPoint.argmax.
    int256 private _bestVal;
    uint256 private _bestIdx;

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
    uint256 private constant SCRATCH_GATE = 3 << 128;

    /// @dev High-water cost of each phase type, remembered across transactions.
    ///      Predicting the next phase from the most expensive one already run cannot
    ///      work: phases differ by an order of magnitude and a transaction that has
    ///      only seen cheap ones will happily start an expensive one it cannot finish.
    ///      Keyed by phase index; ATTEND grows with position, so the stored figure is a
    ///      lower bound and the guard carries margin over it.
    mapping(uint256 => uint256) private _phaseCost;

    /// @dev An ATTEND phase covers the query heads that share one KV head (nHeads/nKv
    ///      under grouped-query attention). Splitting finer would make two phases read
    ///      the same KV slice, and that read is the part that grows with position.

    event GenerationStarted(uint256 promptLen, uint256 maxPos);
    event StepRun(uint256 indexed pos, uint256 indexed layer, uint256 stepIndex, uint256 phase);
    event TokenEmitted(uint256 indexed pos, uint16 token, uint256 stepIndex);
    event GenerationFinished(uint256 tokens, uint256 totalSteps);

    error NotStarted();
    error AlreadyFinished();
    error PromptTooLong();
    error KvOverflow(int256 value);

    constructor(address directory_, Transformer.Config memory c, uint256 macsPerPhase_) {
        directory = directory_;
        macsPerPhase = macsPerPhase_ == 0 ? Transformer.MACS_PER_PHASE : macsPerPhase_;
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
        cursor = 0;
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

        uint256 p = pos;
        uint256 lo = layer;
        // Size the KV cache to this transaction, not to the session: one layer, because
        // every layer is loaded from and stored back to storage individually, and only
        // the positions attention will actually read. EVM memory is quadratic, so the
        // difference between this and `nLayers * maxPos` is worth hundreds of thousands
        // of gas on every transaction — and the saving grows with the requested maxNew.
        // The classifier needs no KV cache and only a shard's worth of logits; a layer
        // phase needs the cache up to this position and no logits at all.
        Transformer.Buffers memory b = lo == c.nLayers
            ? Transformer.newBuffers(c, 1, 1, Transformer.rowsPerPhase(macsPerPhase, c.dim, c.vocab))
            : Transformer.newBuffers(c, p + 1, 1, 1);
        int256[] memory xs = _x;
        for (uint256 i = 0; i < c.dim; ++i) {
            b.resid[i] = xs[i];
        }

        if (lo == c.nLayers) {
            // Inside the prompt the next token is already known, so the classifier is
            // skipped entirely rather than computed and discarded.
            if (p + 1 < promptIds.length) {
                steps += 1;
                emit StepRun(p, lo, steps, 6);
                _emit(p, promptIds[p + 1], true);
                return;
            }
            _classify(chunks, o, c, b, p, reserve);
            return;
        }

        // Embedding a token writes the residual stream, so it has to be persisted like
        // any other write to it. Only W_o and W_2 set this flag otherwise, and a
        // transaction that embeds but ends before reaching one would throw the
        // embedding away — the next transaction reloads zeros and, being past phase 0,
        // never embeds again, so the whole position computes on an empty vector. Small
        // models hid this: their first transaction always reached W_o.
        bool residDirty = false;
        if (lo == 0 && phase == 0 && cursor == 0) {
            uint256 tok0 = p < promptIds.length ? promptIds[p] : generated[p - promptIds.length];
            (bytes memory ew, Transformer.Layout memory eo) = Transformer.sliceEmbed(chunks, o, c, tok0);
            Transformer.embed(ew, eo, c, b, 0, 1);
            residDirty = true;
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
        // The caller's reserve covers its own accounting, but not ours: returning from
        // this loop writes the residual stream back, which is `dim` storage slots and
        // therefore scales with the model. At dim 64 that is ~1.3M gas and a 400k
        // reserve merely looks generous; at dim 256 it is ~5.1M and the transaction
        // runs out after the loop has already decided it had room. The floor is
        // derived from the model rather than trusted from the caller.
        uint256 keep = reserve;
        uint256 floorGas = uint256(c.dim) * 22_000;
        if (keep < floorGas) keep = floorGas;

        uint256 ran = 0;
        while (layer < c.nLayers) {
            uint256 before = gasleft();
            uint256 ph = phase;
            uint256 est = _phaseCost[ph];
            // A phase never run before has no estimate, so it is not started speculatively;
            // it becomes the first phase of the next transaction, where it has the whole
            // budget. Anything that fits in a transaction still runs, and nothing is
            // begun that cannot be finished.
            if (ran > 0 && (est == 0 || before < est + est / 2 + keep)) break;
            if (_phase(chunks, o, c, b, p)) residDirty = true;
            uint256 cost = before - gasleft();
            if (cost > _phaseCost[ph]) _phaseCost[ph] = cost;
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
        emit StepRun(p, layer, steps, phase);
    }

    /// @dev Run one phase of the current layer and advance the cursors.
    ///      Returns true if the residual stream changed. Split into one function per
    ///      phase because each needs its own weight slice, and holding all three sets
    ///      of locals in a single frame runs the compiler out of stack.
    function _phase(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p
    ) private returns (bool residChanged) {
        uint256 ph = phase;
        if (ph == 0) {
            _phaseQ(chunks, o, c, b, p);
            return false;
        }
        if (ph == 1) {
            _phaseKv(chunks, o, c, b, p);
            return false;
        }
        if (ph == 2) {
            _phaseAttend(chunks, o, c, b, p);
            return false;
        }
        if (ph == 3) return _phaseWo(chunks, o, c, b);
        if (ph == 4) {
            _phaseFfn1(chunks, o, c, b);
            return false;
        }
        return _phaseFfn2(chunks, o, c, b);
    }

    /// @dev A shard must be a whole number of heads, so RoPE lands on heads this shard
    ///      actually produced, and a whole number of packed slots, so two shards never
    ///      share one. Aligning to both is aligning to their least common multiple.
    function _headAligned(uint256 n, uint256 hs, uint256 left) private pure returns (uint256) {
        if (n >= left) return left;
        uint256 unit = hs;
        while (unit % PACK_N != 0) {
            unit += hs;
        }
        uint256 a = (n / unit) * unit;
        if (a == 0) a = unit < left ? unit : left;
        return a;
    }

    /// @dev W_q shard. The layer's RMSNorm is taken on the first shard and checkpointed,
    ///      since every shard of Q, K and V consumes the same normalised vector.
    function _phaseQ(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p
    ) private {
        uint256 l = layer;
        uint256 r0 = cursor;
        // Recomputed rather than checkpointed. `normed` is a pure function of the
        // residual stream, which every transaction already loads into memory, so
        // storing it buys nothing and costs a great deal: `dim` slots to write and a
        // cold read of all of them in every transaction that follows. At dim 2048 the
        // write alone is ~10M gas and put the first phase over the ceiling by itself.
        _norm(chunks, o, c, b, l, true);
        uint256 n = _packAligned(Transformer.rowsPerPhase(macsPerPhase, c.dim, c.dim - r0), c.dim - r0);
        (bytes memory w, Transformer.Layout memory s) = Transformer.sliceQRows(chunks, o, c, l, p, r0, n);
        Transformer.prepareQ(w, s, c, b, r0, n);
        _saveVec(0, b.query, r0, n);
        cursor = r0 + n;
        if (cursor == c.dim) {
            phase = 1;
            cursor = 0;
        }
    }

    /// @dev W_k / W_v shard, written straight through to the KV cache in storage
    function _phaseKv(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p
    ) private {
        uint256 l = layer;
        uint256 r0 = cursor;
        _norm(chunks, o, c, b, l, true);
        // W_k and W_v are both computed here, so this phase does twice the work a
        // single-matmul phase does and gets half the row budget.
        uint256 n = _packAligned(Transformer.rowsPerPhase(macsPerPhase / 2, c.dim, c.kvd - r0), c.kvd - r0);
        (bytes memory w, Transformer.Layout memory s) = Transformer.sliceKvRows(chunks, o, c, l, p, r0, n);
        Transformer.prepareKv(w, s, c, b, r0, n);
        _storeKvRange(c, b, l, p, r0, n);
        cursor = r0 + n;
        if (cursor == c.kvd) {
            phase = 2;
            cursor = 0;
        }
    }

    /// @dev ATTEND. Needs no weights at all — only the query and the KV cache — which is
    ///      why the phase that grows with position is also the cheapest to set up.
    function _phaseAttend(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p
    ) private {
        chunks;
        o;
        uint256 l = layer;
        uint256 h0 = cursor;
        uint256 h1 = h0 + c.nHeads / c.nKv;
        if (h1 > c.nHeads) h1 = c.nHeads;
        _loadVec(0, b.query, 0, c.dim);
        // Only this head range's slice of the KV cache is touched, so only that slice
        // is read back — the saving grows with position and is what keeps an ATTEND
        // phase affordable deep into a sequence.
        _loadKvHeads(c, b, l, p, h0, h1);
        Transformer.attend(c, b, l, p, h0, h1);
        _saveVec(SCRATCH_AO, b.attnOut, h0 * c.hs, (h1 - h0) * c.hs);
        cursor = h1;
        if (h1 == c.nHeads) {
            phase = 3;
            cursor = 0;
        }
    }

    /// @dev W_o shard. Output rows are independent, so a row range needs no partial
    ///      sums carried across the boundary — each row lands in the residual stream
    ///      and is finished with. When the last row is done the residual is complete,
    ///      so the FFN's RMSNorm is taken here and checkpointed for the shards ahead.
    function _phaseWo(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b
    ) private returns (bool) {
        uint256 l = layer;
        uint256 r0 = cursor;
        uint256 n = _packAligned(Transformer.rowsPerPhase(macsPerPhase, c.dim, c.dim - r0), c.dim - r0);
        _loadVec(SCRATCH_AO, b.attnOut, 0, c.dim);
        (bytes memory w, Transformer.Layout memory s) = Transformer.sliceWo(chunks, o, c, l, r0, n);
        Transformer.projectWo(w, s, c, b, r0, n);
        for (uint256 i = r0; i < r0 + n; ++i) {
            b.resid[i] += b.proj[i];
        }
        cursor = r0 + n;
        if (cursor == c.dim) {
            phase = 4;
            cursor = 0;
        }
        return true;
    }

    /// @dev SwiGLU shard: W_1 and W_3 rows, combined and stored as `gate`
    function _phaseFfn1(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b
    ) private {
        uint256 l = layer;
        uint256 r0 = cursor;
        // W_1 and W_3 are both computed here, so half the row budget as above.
        uint256 n = _packAligned(Transformer.rowsPerPhase(macsPerPhase / 2, c.dim, c.hidden - r0), c.hidden - r0);
        _norm(chunks, o, c, b, l, false);
        (bytes memory w, Transformer.Layout memory s) = Transformer.sliceFfn1(chunks, o, c, l, r0, n);
        Transformer.projectFfn1(w, s, c, b, r0, n);
        _saveVec(SCRATCH_GATE, b.gate, r0, n);
        cursor = r0 + n;
        if (cursor == c.hidden) {
            phase = 5;
            cursor = 0;
        }
    }

    /// @dev W_2 shard, back into the residual stream. Row i of W_2 reads the whole of
    ///      `gate`, so every shard loads all of it — the one place this decomposition
    ///      pays an O(hidden) read per phase rather than a slice.
    function _phaseFfn2(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b
    ) private returns (bool) {
        uint256 l = layer;
        uint256 r0 = cursor;
        uint256 n = _packAligned(Transformer.rowsPerPhase(macsPerPhase, c.hidden, c.dim - r0), c.dim - r0);
        _loadVec(SCRATCH_GATE, b.gate, 0, c.hidden);
        (bytes memory w, Transformer.Layout memory s) = Transformer.sliceFfn2(chunks, o, c, l, r0, n);
        Transformer.projectFfn2(w, s, c, b, r0, n);
        for (uint256 i = r0; i < r0 + n; ++i) {
            b.resid[i] += b.proj[i];
        }
        cursor = r0 + n;
        if (cursor == c.dim) {
            phase = 0;
            cursor = 0;
            layer = l + 1;
        }
        return true;
    }

    /// @dev RMSNorm the residual stream into `b.normed`, with the attention gains or
    ///      the feed-forward ones. Cheap next to a matmul shard and far cheaper than
    ///      keeping the result in storage between phases.
    function _norm(
        address[] memory pages,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 l,
        bool att
    ) private view {
        if (att) {
            (bytes memory w, Transformer.Layout memory s) = Transformer.sliceRmsAtt(pages, o, c, l);
            Transformer.rmsnorm(w, s.rmsAtt, 0, b.resid, b.normed, c.dim);
        } else {
            (bytes memory w, Transformer.Layout memory s) = Transformer.sliceRmsFfn(pages, o, c, l);
            Transformer.rmsnorm(w, s.rmsFfn, 0, b.resid, b.normed, c.dim);
        }
    }

    /// @dev Save `len` values of `src` from `off`, packed four int64 lanes per slot.
    ///      `off` and `len` are multiples of PACK_N for every caller here (head slices
    ///      are `hs` wide, and `hs` is a multiple of four).
    function _saveVec(uint256 space, int256[] memory src, uint256 off, uint256 len) private {
        for (uint256 i = 0; i < len; i += PACK_N) {
            uint256 n = len - i < PACK_N ? len - i : PACK_N;
            _scratch[space + (off + i) / PACK_N] = _pack(src, off + i, n);
        }
    }

    function _loadVec(uint256 space, int256[] memory dst, uint256 off, uint256 len) private view {
        for (uint256 i = 0; i < len; i += PACK_N) {
            uint256 packed = _scratch[space + (off + i) / PACK_N];
            uint256 n = len - i < PACK_N ? len - i : PACK_N;
            for (uint256 j = 0; j < n; ++j) {
                dst[off + i + j] = _lane(packed, j);
            }
        }
    }

    /// @dev Round a row chunk down to a whole number of packed slots so two chunks can
    ///      never land in the same one; the final chunk takes whatever remains, and its
    ///      slots are exclusively its own because every chunk before it was aligned.
    function _packAligned(uint256 n, uint256 left) private pure returns (uint256) {
        if (n >= left) return left;
        uint256 a = (n / PACK_N) * PACK_N;
        if (a == 0) a = PACK_N < left ? PACK_N : left;
        return a;
    }

    /// @dev The classifier, sharded over the vocabulary. It is the largest single
    ///      matmul in the model — rows equal to the vocabulary, each `dim` wide — so at
    ///      any realistic vocabulary it is the first thing that cannot fit in one
    ///      transaction. At 32k rows and dim 2048 it is roughly 3.2 BILLION gas whole.
    ///
    ///      Rows are independent, so shards need no partial sums. What they do need is
    ///      the argmax, and carrying that as a running best value and index costs two
    ///      slots instead of a logit vector the width of the vocabulary.
    function _classify(
        address[] memory chunks,
        Transformer.Layout memory o,
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 p,
        uint256 reserve
    ) private {
        if (cursor == 0) {
            _bestVal = type(int256).min;
            _bestIdx = 0;
        }
        {
            (bytes memory rw, Transformer.Layout memory ro) = Transformer.sliceRmsFinal(chunks, o, c);
            Transformer.rmsnorm(rw, ro.rmsFinal, 0, b.resid, b.normed, c.dim);
        }

        int256 bestVal = _bestVal;
        uint256 bestIdx = _bestIdx;
        uint256 ran = 0;
        uint256 keep = reserve < 200_000 ? 200_000 : reserve;

        while (cursor < c.vocab) {
            uint256 before = gasleft();
            uint256 est = _phaseCost[6];
            if (ran > 0 && (est == 0 || before < est + est / 2 + keep)) break;
            uint256 r0 = cursor;
            uint256 n = Transformer.rowsPerPhase(macsPerPhase, c.dim, c.vocab - r0);
            (bytes memory w, Transformer.Layout memory s) = Transformer.sliceClassifyRows(chunks, o, c, r0, n);
            Transformer.classifyShard(w, s, c, b, n);
            for (uint256 i = 0; i < n; ++i) {
                if (b.logits[i] > bestVal) {
                    bestVal = b.logits[i];
                    bestIdx = r0 + i;
                }
            }
            cursor = r0 + n;
            uint256 cost = before - gasleft();
            if (cost > _phaseCost[6]) _phaseCost[6] = cost;
            unchecked {
                ran += 1;
            }
        }

        steps += ran;
        _bestVal = bestVal;
        _bestIdx = bestIdx;
        emit StepRun(p, c.nLayers, steps, 6);
        if (cursor == c.vocab) _emit(p, bestIdx, false);
    }

    /// @dev Pick the next token (or teacher-force from the prompt) and advance
    function _emit(uint256 p, uint256 next, bool inPrompt) private {
        bool stop = false;
        if (!inPrompt) {
            generated.push(uint16(next));
            emit TokenEmitted(p, uint16(next), steps);
            if (next == Decoder.BOS || next == Decoder.EOS) stop = true;
        }
        layer = 0;
        phase = 0;
        cursor = 0;
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

    /// @notice The residual stream as it currently stands, for comparing a partially
    ///         advanced run against the reference implementation.
    function residual() external view returns (int256[] memory) {
        return _x;
    }

    function generatedIds() external view returns (uint16[] memory) {
        return generated;
    }

    /// @notice How many phases one position costs, for this model and this MAC budget.
    /// @dev Clients need it to size funding and batches, and it must not be a constant
    ///      they carry: it depends on the layer count, the KV head count, the tensor
    ///      widths and the budget, so a different model or a different deployment gives
    ///      a different answer. Deriving it here is what keeps a client model-agnostic.
    function phasesPerPosition() public view returns (uint256) {
        Transformer.Config memory c = config();
        uint256 perLayer = _chunkCount(c.dim, c.dim) // W_q
            + _chunkCount2(c.kvd, c.dim) // W_k / W_v, two matmuls per phase
            + c.nKv // ATTEND, one phase per KV head
            + _chunkCount(c.dim, c.dim) // W_o
            + _chunkCount2(c.hidden, c.dim) // SwiGLU, two matmuls per phase
            + _chunkCount(c.dim, c.hidden); // W_2
        // The classifier is itself sharded over the vocabulary
        return perLayer * c.nLayers + _chunkCount(c.vocab, c.dim);
    }

    /// @dev As `_chunkCount`, for phases that run two matmuls and take half the budget
    function _chunkCount2(uint256 rows, uint256 cols) private view returns (uint256 n) {
        uint256 r = 0;
        while (r < rows) {
            r += _packAligned(Transformer.rowsPerPhase(macsPerPhase / 2, cols, rows - r), rows - r);
            unchecked {
                n += 1;
            }
        }
    }

    /// @dev Walk the same cursor the phases walk, so the count matches exactly
    function _chunkCount(uint256 rows, uint256 cols) private view returns (uint256 n) {
        uint256 r = 0;
        while (r < rows) {
            r += _packAligned(Transformer.rowsPerPhase(macsPerPhase, cols, rows - r), rows - r);
            unchecked {
                n += 1;
            }
        }
    }

    function progress()
        external
        view
        returns (uint256 pos_, uint256 layer_, uint256 maxPos_, bool done_, uint256 steps_)
    {
        return (pos, layer, maxPos, finished, steps);
    }

    // ------------------------------------------------------------- internals
    /// @dev The directory lists page contracts, each holding up to PER_PAGE chunk
    ///      addresses, because listing chunks directly in one directory caps the model
    ///      at 1228 chunks — 30MB, or 15M int16 parameters. Pages square that.
    ///
    ///      The chunk list is expanded here rather than resolved per lookup, costing one
    ///      word of memory per chunk: negligible at thousands of chunks, and the
    ///      dominant per-transaction cost in the tens of thousands, which is where the
    ///      next change would be needed.
    function _dir() private view returns (address tokPtr, address[] memory chunks) {
        bytes memory d = WeightStore.read(directory);
        uint256 np = d.length / 20 - 1;
        tokPtr = _addr(d, 0);

        bytes[] memory raw = new bytes[](np);
        uint256 total = 0;
        for (uint256 i = 0; i < np; ++i) {
            raw[i] = WeightStore.read(_addr(d, (i + 1) * 20));
            total += raw[i].length / 20;
        }
        chunks = new address[](total);
        uint256 k = 0;
        for (uint256 i = 0; i < np; ++i) {
            uint256 m = raw[i].length / 20;
            for (uint256 j = 0; j < m; ++j) {
                chunks[k] = _addr(raw[i], j * 20);
                unchecked {
                    k += 1;
                }
            }
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

    /// @dev Write K/V lanes [r0, r0+n) of this position's row. Shards write their own
    ///      slice straight through, so the full vector never has to exist at once.
    function _storeKvRange(
        Transformer.Config memory c,
        Transformer.Buffers memory b,
        uint256 l,
        uint256 p,
        uint256 r0,
        uint256 n
    ) private {
        for (uint256 i = r0; i < r0 + n; i += PACK_N) {
            uint256 k = r0 + n - i < PACK_N ? r0 + n - i : PACK_N;
            _kv[_key(c, false, l, p, i / PACK_N)] = _pack(b.key, i, k);
            _kv[_key(c, true, l, p, i / PACK_N)] = _pack(b.value, i, k);
        }
    }
}
