// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {FixedPoint} from "../src/FixedPoint.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice Spreading inference across transactions is only legitimate if it changes
///         nothing, so the resumable contract must reproduce the specification's token
///         sequence exactly.
contract ResumableTest is Test {
    using stdJson for string;

    OnchainLLM internal llm;
    address internal dirPtr;
    Transformer.Config internal cfg;

    function setUp() public {
        _deploy(0);
    }

    function _deploy(uint256 macs) internal {
        string memory root = vm.projectRoot();
        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));

        // token table: [u16 count][u8 maxLen][u16 offset per token][bytes]
        bytes memory table =
            vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/tokens.hex"))));
        address tokPtr = _writePaged(table);

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

        Transformer.Config memory c = Transformer.Config({
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
            invSqrtHs: 0,
            nExperts: 0,
            topK: 0
        });
        dirPtr = WeightStore.write(dir);
        cfg = c;
        llm = new OnchainLLM(dirPtr, c, macs);
    }

    function test_ResumableMatchesSpec() public {
        string memory spec = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/generation_vectors.json"));
        uint256[] memory promptRaw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");

        uint16[] memory prompt = new uint16[](promptRaw.length);
        for (uint256 i = 0; i < promptRaw.length; ++i) {
            prompt[i] = uint16(promptRaw[i]);
        }

        llm.start(prompt, want.length);
        uint256 guard;
        while (!llm.finished()) {
            llm.step(400_000);
            guard++;
            require(guard < 500, "step budget exceeded");
        }

        uint16[] memory got = llm.generatedIds();
        assertEq(got.length, want.length, "token count differs");
        for (uint256 i = 0; i < want.length; ++i) {
            assertEq(uint256(got[i]), want[i], "resumable diverged from spec");
        }
        console2.log("tokens matching spec:", want.length);
        console2.log("transactions:", llm.steps());
        console2.log("decoded text:", llm.text());
    }

    /// @notice Sharding a matmul by output row must not change its result. With the
    ///         default budget this model never actually shards — every tensor fits in
    ///         one phase — so the split path would otherwise go untested until a model
    ///         large enough to need it. A deliberately tiny budget forces ten-row
    ///         chunks of a 64-wide tensor and four-row chunks of a 172-wide one.
    function test_RowShardingMatchesUnsharded() public {
        string memory spec = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/generation_vectors.json"));
        uint256[] memory promptRaw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");
        uint16[] memory prompt = new uint16[](promptRaw.length);
        for (uint256 i = 0; i < promptRaw.length; ++i) {
            prompt[i] = uint16(promptRaw[i]);
        }

        _deploy(700);
        llm.start(prompt, want.length);
        uint256 guard;
        while (!llm.finished()) {
            llm.step(400_000);
            guard++;
            require(guard < 4000, "step budget exceeded");
        }

        uint16[] memory got = llm.generatedIds();
        assertEq(got.length, want.length, "token count differs when sharded");
        for (uint256 i = 0; i < want.length; ++i) {
            assertEq(uint256(got[i]), want[i], "row sharding changed the output");
        }
        console2.log("sharded transactions:", llm.steps());
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
