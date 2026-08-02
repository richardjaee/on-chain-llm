// SPDX-License-Identifier: MIT
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

    function setUp() public {
        string memory root = vm.projectRoot();
        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));

        // token table: [u16 count][u8 maxLen][u16 offset per token][bytes]
        bytes memory table = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/tokens.hex"))));
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
            dir = abi.encodePacked(dir, WeightStore.write(part));
        }

        Transformer.Config memory c = Transformer.Config({
            dim: 64, hidden: 172, nLayers: 5, nHeads: 8, nKv: 4,
            vocab: 512, seqLen: 512, hs: 8, kvd: 32, shared: true, invSqrtHs: 0
        });
        llm = new OnchainLLM(WeightStore.write(dir), c);
    }

    function test_ResumableMatchesSpec() public {
        string memory spec = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/generation_vectors.json"));
        uint256[] memory promptRaw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");

        uint16[] memory prompt = new uint16[](promptRaw.length);
        for (uint256 i = 0; i < promptRaw.length; ++i) prompt[i] = uint16(promptRaw[i]);

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
}
