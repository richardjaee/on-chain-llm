// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";
import {FixedPoint} from "../src/FixedPoint.sol";

/// @notice Does the sharded machinery hold up at a width no real checkpoint here
///         reaches? Gas does not depend on the values being multiplied, only on the
///         shapes and the memory and storage touched, so a synthetic blob of the right
///         shape measures the real thing. tools/synth.py generates it.
///
///         Run with:  SYNTH=/path/to/synth.hex forge test --match-path test/Scale.t.sol -vv
contract ScaleTest is Test {
    using stdJson for string;

    OnchainLLM internal llm;
    Transformer.Config internal cfg;
    bool internal have;
    address[] internal chunkAddrs;
    uint256 internal CAP;

    function setUp() public {
        string memory path = vm.envOr("SYNTH", string(""));
        if (bytes(path).length == 0) return;
        have = true;
        CAP = vm.envOr("CAP", uint256(14_000_000));

        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(path)));
        // A real token table when one is supplied, so `text()` can be checked too.
        string memory tp = vm.envOr("TOKENS", string(""));
        address tokPtr;
        if (bytes(tp).length > 0) {
            bytes memory tbl = vm.parseBytes(string.concat("0x", vm.readFile(tp)));
            tokPtr = _writePaged(tbl);
        } else {
            tokPtr = _writePaged(hex"00010000");
        }

        // Chunk addresses go into page contracts, and the directory lists the pages.
        // A single directory would cap the model at MAX_PAYLOAD / 20 chunks.
        bytes memory page;
        bytes memory dir = abi.encodePacked(tokPtr);
        uint256 n = (blob.length + WeightStore.MAX_PAYLOAD - 1) / WeightStore.MAX_PAYLOAD;
        for (uint256 i = 0; i < n; ++i) {
            uint256 s = i * WeightStore.MAX_PAYLOAD;
            uint256 len = blob.length - s;
            if (len > WeightStore.MAX_PAYLOAD) len = WeightStore.MAX_PAYLOAD;
            bytes memory part = new bytes(len);
            assembly ("memory-safe") {
                let src := add(add(blob, 0x20), s)
                let dst := add(part, 0x20)
                for { let k := 0 } lt(k, len) { k := add(k, 0x20) } { mstore(add(dst, k), mload(add(src, k))) }
            }
            address cp = WeightStore.write(part);
            chunkAddrs.push(cp);
            page = abi.encodePacked(page, cp);
            if (page.length / 20 == WeightStore.PER_PAGE) {
                dir = abi.encodePacked(dir, WeightStore.write(page));
                page = "";
            }
        }
        if (page.length > 0) dir = abi.encodePacked(dir, WeightStore.write(page));

        console2.log("chunks deployed:", n);

        uint32 dim = uint32(vm.envOr("DIM", uint256(256)));
        uint32 heads = uint32(vm.envOr("HEADS", uint256(8)));
        uint32 kv = uint32(vm.envOr("KVHEADS", uint256(4)));
        cfg = Transformer.Config({
            dim: dim,
            hidden: uint32(vm.envOr("HIDDEN", uint256(688))),
            nLayers: uint32(vm.envOr("LAYERS", uint256(2))),
            nHeads: heads,
            nKv: kv,
            vocab: uint32(vm.envOr("VOCAB", uint256(512))),
            seqLen: uint32(vm.envOr("SEQLEN", uint256(256))),
            hs: dim / heads,
            kvd: (dim / heads) * kv,
            shared: true,
            invSqrtHs: 0
        });
        console2.log("dim", cfg.dim, "hidden", cfg.hidden);
        llm = new OnchainLLM(WeightStore.write(dir), cfg, vm.envOr("MACS", uint256(0)));
        console2.log("macsPerPhase", llm.macsPerPhase());
    }

    /// @notice Cost of each phase in isolation. A large `reserve` makes the gas loop
    ///         stop after the one phase it always runs unconditionally, so each line is
    ///         a single phase rather than however many happened to fit.
    function test_PhaseCosts() public {
        if (!have) return;
        uint16[] memory prompt = new uint16[](3);
        prompt[0] = 1;
        prompt[1] = 7;
        prompt[2] = 9;
        llm.start(prompt, 4);
        for (uint256 i = 0; i < 30; ++i) {
            if (llm.finished()) break;
            uint256 ph = llm.phase();
            uint256 cur = llm.cursor();
            uint256 l = llm.layer();
            uint256 g = gasleft();
            llm.step{gas: 250_000_000}(249_000_000);
            console2.log("  layer", l, "phase", ph);
            console2.log("     cursor", cur, "gas", g - gasleft());
        }
    }

    /// @notice Whole-blob path only: does the library itself get the right first token?
    ///         Isolates the arithmetic from the resumable phase machine.
    function test_LibraryFirstToken() public view {
        if (!have) return;
        Transformer.Config memory c = llm.config();
        Transformer.Layout memory o = Transformer.layout(c);
        uint256 blobLen = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;
        console2.log("blob len from layout:", blobLen);

        bytes memory w = new bytes(blobLen);
        WeightStore.readRange(chunkAddrs, 0, blobLen, w);
        Transformer.Buffers memory b = Transformer.newBuffers(c, 4);
        Transformer.forwardRange(w, o, c, b, 1, 0, 0, c.nLayers);
        Transformer.normFinal(w, o, c, b);
        Transformer.classifyRange(w, o, c, b, 0, c.vocab);
        console2.log("argmax at pos 0:", FixedPoint.argmax(b.logits));
        console2.log("logits[0..2]:", uint256(b.logits[0] < 0 ? -b.logits[0] : b.logits[0]));
    }

    /// @notice Does the single-row embedding slice produce the same vector as reading
    ///         the embedding out of the whole blob?
    function test_EmbedSliceMatches() public view {
        if (!have) return;
        Transformer.Config memory c = llm.config();
        Transformer.Layout memory o = Transformer.layout(c);
        uint256 blobLen = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;

        bytes memory w = new bytes(blobLen);
        WeightStore.readRange(chunkAddrs, 0, blobLen, w);
        Transformer.Buffers memory full = Transformer.newBuffers(c, 1);
        Transformer.embed(w, o, c, full, 1, c.vocab);

        (bytes memory ew, Transformer.Layout memory eo) = Transformer.sliceEmbed(chunkAddrs, o, c, 1);
        Transformer.Buffers memory sl = Transformer.newBuffers(c, 1);
        Transformer.embed(ew, eo, c, sl, 0, 1);

        uint256 bad;
        for (uint256 i = 0; i < c.dim; ++i) {
            if (full.resid[i] != sl.resid[i]) bad++;
        }
        console2.log("full[0]  :", full.resid[0]);
        console2.log("slice[0] :", sl.resid[0]);
        console2.log("embed mismatches:", bad, "of", c.dim);
    }

    /// @notice Run the resumable path through one position's layers, then compare its
    ///         residual stream against the library's. Isolates which side is wrong.
    function test_ResidualMatchesLibrary() public {
        if (!have) return;
        Transformer.Config memory c = llm.config();
        Transformer.Layout memory o = Transformer.layout(c);

        uint16[] memory prompt = new uint16[](1);
        prompt[0] = 1;
        llm.start(prompt, 1);
        // advance only until the classifier stage is reached
        uint256 guard;
        uint256 upto = vm.envOr("UPTO", uint256(c.nLayers));
        while (llm.layer() < upto && guard < 100000) {
            llm.step{gas: CAP}(400_000);
            guard++;
        }
        int256[] memory onchain = llm.residual();
        console2.log("phases run:", llm.steps(), "layer:", llm.layer());

        uint256 blobLen = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;
        bytes memory w = new bytes(blobLen);
        WeightStore.readRange(chunkAddrs, 0, blobLen, w);
        Transformer.Buffers memory b = Transformer.newBuffers(c, 2);
        Transformer.forwardRange(w, o, c, b, 1, 0, 0, upto);

        uint256 bad;
        for (uint256 i = 0; i < c.dim; ++i) {
            if (onchain[i] != b.resid[i]) {
                if (bad < 4) {
                    console2.log("  mismatch at", i);
                    console2.log("    onchain", onchain[i]);
                    console2.log("    library", b.resid[i]);
                }
                bad++;
            }
        }
        console2.log("mismatched elements:", bad, "of", c.dim);
    }

    /// @notice A real trained checkpoint, end to end, against the Python specification.
    ///         Synthetic blobs prove the shapes are affordable; only a trained one proves
    ///         the arithmetic still means something at this size.
    function test_MatchesVectors() public {
        if (!have) return;
        string memory vf = vm.envOr("VECTORS", string(""));
        if (bytes(vf).length == 0) {
            console2.log("VECTORS not set - skipping");
            return;
        }
        string memory spec = vm.readFile(vf);
        uint256[] memory praw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");
        uint256 n = vm.envOr("MAXNEW", want.length);
        if (n > want.length) n = want.length;

        uint16[] memory prompt = new uint16[](praw.length);
        for (uint256 i = 0; i < praw.length; ++i) {
            prompt[i] = uint16(praw[i]);
        }
        llm.start(prompt, n);
        console2.log("phasesPerPosition:", llm.phasesPerPosition());

        uint256 txs;
        uint256 total;
        uint256 peak;
        while (!llm.finished() && txs < 200000) {
            uint256 g = gasleft();
            llm.step{gas: CAP}(400_000);
            uint256 used = g - gasleft();
            total += used;
            if (used > peak) peak = used;
            txs++;
        }
        uint16[] memory got = llm.generatedIds();
        (uint256 gp, uint256 gl, uint256 gm, bool gd, uint256 gs) = llm.progress();
        console2.log("finished/pos/maxPos:", gd, gp, gm);
        console2.log("layer/steps        :", gl, gs);
        console2.log("generated count    :", got.length);
        for (uint256 i = 0; i < got.length && i < 6; ++i) {
            console2.log("  got", i, uint256(got[i]));
        }
        for (uint256 i = 0; i < n; ++i) {
            assertEq(uint256(got[i]), want[i], "on-chain diverged from the specification");
        }
        console2.log("tokens matching spec:", n);
        console2.log("decoded text:", llm.text());
        console2.log("transactions        :", txs);
        console2.log("total gas           :", total);
        console2.log("peak tx gas         :", peak);
    }

    function test_WideModelRuns() public {
        CAP = vm.envOr("CAP", uint256(14_000_000));
        if (!have) {
            console2.log("SYNTH not set - skipping");
            return;
        }
        console2.log("phasesPerPosition:", llm.phasesPerPosition());

        uint16[] memory prompt = new uint16[](3);
        prompt[0] = 1;
        prompt[1] = 7;
        prompt[2] = 9;
        llm.start(prompt, 4);

        uint256 txs;
        uint256 total;
        uint256 peak;
        uint256 lastPos;
        while (!llm.finished() && txs < 20000) {
            (uint256 p,,,,) = llm.progress();
            uint256 ph = llm.phase();
            uint256 cur = llm.cursor();
            uint256 g = gasleft();
            try llm.step{gas: CAP}(400_000) {
                uint256 used = g - gasleft();
                total += used;
                if (used > peak) peak = used;
                txs++;
                if (p != lastPos) {
                    console2.log("  position", p, "cumulative tx", txs);
                    lastPos = p;
                }
            } catch (bytes memory err) {
                console2.log("STOPPED at position", p);
                console2.log("  layer/phase/cursor:", llm.layer(), ph, cur);
                console2.logBytes(err);
                return;
            }
        }
        console2.log("transactions :", txs);
        console2.log("total gas    :", total);
        console2.log("peak tx gas  :", peak);
        console2.log("steps        :", llm.steps());

        assertLt(peak, CAP + 1, "a transaction exceeded the ceiling");
    }

    /// @dev Store a blob as chunk contracts and return a page listing them
    function _writePaged(bytes memory blob) internal returns (address) {
        bytes memory list;
        uint256 n = (blob.length + WeightStore.MAX_PAYLOAD - 1) / WeightStore.MAX_PAYLOAD;
        for (uint256 i = 0; i < n; ++i) {
            uint256 s0 = i * WeightStore.MAX_PAYLOAD;
            uint256 len = blob.length - s0;
            if (len > WeightStore.MAX_PAYLOAD) len = WeightStore.MAX_PAYLOAD;
            bytes memory part = new bytes(len);
            assembly ("memory-safe") {
                let src := add(add(blob, 0x20), s0)
                let dst := add(part, 0x20)
                for { let k := 0 } lt(k, len) { k := add(k, 0x20) } { mstore(add(dst, k), mload(add(src, k))) }
            }
            list = abi.encodePacked(list, WeightStore.write(part));
        }
        return WeightStore.write(list);
    }
}
