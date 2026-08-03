// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice Temporary instrumentation: where does the gas in one `step` actually go?
contract ProfileTest is Test {
    OnchainLLM internal llm;
    address[] internal chunks;
    Transformer.Config internal cfg;

    function setUp() public {
        string memory root = vm.projectRoot();
        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));
        bytes memory table =
            vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/tokens.hex"))));
        address tokPtr = WeightStore.write(table);

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
            address p = WeightStore.write(part);
            chunks.push(p);
            dir = abi.encodePacked(dir, p);
        }

        cfg = Transformer.Config({
            dim: 64,
            hidden: 172,
            nLayers: 5,
            nHeads: 8,
            nKv: 4,
            vocab: 512,
            seqLen: 512,
            hs: 8,
            kvd: 32,
            shared: true,
            invSqrtHs: 0
        });
        llm = new OnchainLLM(WeightStore.write(dir), cfg);
    }

    function test_Profile() public {
        Transformer.Config memory c = llm.config();
        Transformer.Layout memory o = Transformer.layout(c);
        uint256 blobLen = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;
        console2.log("blob bytes", blobLen);

        address[] memory ch = chunks;

        uint256 g = gasleft();
        bytes memory w = new bytes(blobLen);
        WeightStore.readRange(ch, 0, blobLen, w);
        console2.log("A. read WHOLE blob     ", g - gasleft());

        // What one layer actually needs: its slices of wq/wk/wv/wo/w1/w2/w3 + norms
        uint256 dim = c.dim;
        uint256 oneLayer = Transformer.blockSize(dim, dim) // wq slice
            + Transformer.blockSize(c.kvd, dim) * 2 // wk, wv
            + Transformer.blockSize(dim, dim) // wo
            + Transformer.blockSize(c.hidden, dim) * 2 // w1, w3
            + Transformer.blockSize(dim, c.hidden); // w2
        console2.log("   one layer needs bytes", oneLayer);

        g = gasleft();
        bytes memory slice = new bytes(oneLayer);
        WeightStore.readRange(ch, 0, oneLayer, slice);
        console2.log("B. read ONE-LAYER slice", g - gasleft());

        uint256 maxPos = 18;
        g = gasleft();
        Transformer.Buffers memory b = Transformer.newBuffers(c, maxPos);
        console2.log("C. newBuffers(maxPos=18)", g - gasleft());
        b;

        g = gasleft();
        Transformer.Buffers memory b2 = Transformer.newBuffers(c, 1);
        console2.log("   newBuffers(maxPos=1) ", g - gasleft());
        b2;

        // Full end-to-end step costs across a generation
        uint16[] memory prompt = new uint16[](5);
        prompt[0] = 1;
        prompt[1] = 320;
        prompt[2] = 327;
        prompt[3] = 261;
        prompt[4] = 276;
        llm.start(prompt, 8);

        // Cap each call at 14M so the batching loop behaves as it does on chain,
        // where the provider ceiling — not Foundry's unlimited budget — decides how
        // many layers fit in one transaction.
        uint256 total;
        uint256 i;
        uint256 peak;
        while (!llm.finished() && i < 300) {
            uint256 s = gasleft();
            llm.step{gas: 14_000_000}(400_000);
            uint256 used = s - gasleft();
            total += used;
            if (used > peak) peak = used;
            if (i < 8) console2.log("   step", i, used);
            i++;
        }
        console2.log("D. steps                ", i);
        console2.log("D. total gas            ", total);
        console2.log("D. peak step            ", peak);
        console2.log("D. mean step            ", total / i);

        // The benchmark is only meaningful if the run actually completed under a cap
        // no transaction is allowed to exceed.
        assertTrue(llm.finished(), "generation did not finish under a 14M cap");
        assertLt(peak, 14_000_000, "a step exceeded the per-transaction ceiling");
    }

    /// @notice Find the position at which a single layer no longer fits in one
    ///         transaction. Past this point the run strands: a layer is already the
    ///         smallest unit `step` can execute, so there is nothing left to split.
    function test_FindStrandPosition() public {
        uint16[] memory prompt = new uint16[](5);
        prompt[0] = 1;
        prompt[1] = 320;
        prompt[2] = 327;
        prompt[3] = 261;
        prompt[4] = 276;
        llm.start(prompt, 120);

        uint256 lastPos;
        for (uint256 i = 0; i < 2000; ++i) {
            (uint256 p,,, bool done,) = llm.progress();
            if (done) break;
            uint256 g = gasleft();
            try llm.step{gas: 14_000_000}(400_000) {
                uint256 used = g - gasleft();
                if (p != lastPos) {
                    console2.log("pos", p, "step gas", used);
                    lastPos = p;
                }
            } catch {
                console2.log("STRANDED at position", p);
                console2.log("  a single layer no longer fits under 14,000,000 gas");
                return;
            }
        }
        console2.log("completed without stranding");
    }

    /// @notice Cost of ONE layer's compute, isolated from setup and storage
    function test_ProfileOneLayer() public view {
        Transformer.Config memory c = llm.config();
        Transformer.Layout memory o = Transformer.layout(c);
        uint256 blobLen = o.ropeSin + uint256(c.seqLen) * (c.hs / 2) * 4;
        address[] memory ch = chunks;

        bytes memory w = new bytes(blobLen);
        WeightStore.readRange(ch, 0, blobLen, w);
        Transformer.Buffers memory b = Transformer.newBuffers(c, 18);

        uint256 g = gasleft();
        Transformer.forwardRange(w, o, c, b, 320, 0, 0, 1);
        console2.log("E. forwardRange 1 layer @pos0", g - gasleft());

        g = gasleft();
        Transformer.forwardRange(w, o, c, b, 320, 1, 0, 1);
        console2.log("E. forwardRange 1 layer @pos1", g - gasleft());

        g = gasleft();
        Transformer.normFinal(w, o, c, b);
        Transformer.classifyRange(w, o, c, b, 0, c.vocab);
        console2.log("F. classifier (vocab 512)    ", g - gasleft());

        // matmul cost alone, for the largest tensor in a layer
        int256[] memory x = new int256[](c.dim);
        int256[] memory out = new int256[](c.hidden);
        g = gasleft();
        Transformer.matmul(w, o.w1, uint256(c.nLayers) * c.hidden, 0, c.hidden, c.dim, x, out, 0);
        console2.log("G. matmul w1 (172x64)        ", g - gasleft());
    }
}
