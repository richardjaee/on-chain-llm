// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice Does the sharded machinery hold up at a width no real checkpoint here
///         reaches? Gas does not depend on the values being multiplied, only on the
///         shapes and the memory and storage touched, so a synthetic blob of the right
///         shape measures the real thing. tools/synth.py generates it.
///
///         Run with:  SYNTH=/path/to/synth.hex forge test --match-path test/Scale.t.sol -vv
contract ScaleTest is Test {
    OnchainLLM internal llm;
    Transformer.Config internal cfg;
    bool internal have;

    function setUp() public {
        string memory path = vm.envOr("SYNTH", string(""));
        if (bytes(path).length == 0) return;
        have = true;

        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(path)));
        // The token table is only needed to decode output, which this never does.
        address tokPtr = WeightStore.write(hex"00010000");

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
            page = abi.encodePacked(page, WeightStore.write(part));
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

    uint256 internal CAP;

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
}
