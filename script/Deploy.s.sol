// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice Deploys the model: token table and weight chunks as data contracts, a
///         directory listing them, then the resumable inference contract.
contract DeployScript is Script {
    OnchainLLM public llm;

    function run() public {
        string memory root = vm.projectRoot();
        bytes memory blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));
        bytes memory table =
            vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/tokens.hex"))));

        vm.startBroadcast();

        address tokPtr = WeightStore.write(table);
        bytes memory dir = abi.encodePacked(tokPtr);
        uint256 n = (blob.length + WeightStore.MAX_PAYLOAD - 1) / WeightStore.MAX_PAYLOAD;
        for (uint256 i = 0; i < n; ++i) {
            uint256 s = i * WeightStore.MAX_PAYLOAD;
            uint256 len = blob.length - s;
            if (len > WeightStore.MAX_PAYLOAD) len = WeightStore.MAX_PAYLOAD;
            dir = abi.encodePacked(dir, WeightStore.write(_slice(blob, s, len)));
        }
        address dirPtr = WeightStore.write(dir);

        llm = new OnchainLLM(
            dirPtr,
            Transformer.Config({
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
            }),
            0 // default MACs per sharded matmul phase
        );

        vm.stopBroadcast();

        console.log("token table :", tokPtr);
        console.log("weight chunks:", n);
        console.log("directory   :", dirPtr);
        console.log("OnchainLLM  :", address(llm));
        console.log("DEPLOYED_TARGET=%s", address(llm));
    }

    function _slice(bytes memory d, uint256 s, uint256 len) internal pure returns (bytes memory out) {
        out = new bytes(len);
        assembly ("memory-safe") {
            let src := add(add(d, 0x20), s)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, len) { i := add(i, 0x20) } { mstore(add(dst, i), mload(add(src, i))) }
        }
    }
}
