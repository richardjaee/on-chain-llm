// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {FixedPoint} from "../src/FixedPoint.sol";
import {OnchainLLM} from "../src/OnchainLLM.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice The first time the mixture dispatch runs.
///
/// @dev Everything about the mixture has been checked in pieces: the routing against three
///      implementations, the layout's offsets against a dense blob, the dispatch against
///      the compiler. None of that executes `_phaseRoute`. This does.
///
///      The weights are synthetic, which is sufficient for what is under test. Whether the
///      router picks the experts the specification picks, whether their rows are read from
///      the right offsets, and whether the phase ladder walks them in the right order are
///      properties of the dispatch and not of the values — and a wrong answer on random
///      weights is still a wrong answer.
///
///      What this does not show is that a real mixture produces good text. That needs a
///      checkpoint and an exporter, and is a separate claim.
contract MixtureRunTest is Test {
    using stdJson for string;

    OnchainLLM internal llm;
    Transformer.Config internal cfg;
    string internal spec;

    function setUp() public {
        string memory root = vm.projectRoot();
        spec = vm.readFile(string.concat(root, "/artifacts/moe_run.json"));
        bytes memory blob =
            vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/moe_synth.hex"))));
        bytes memory table =
            vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/tokens.hex"))));

        address tokPtr = _writePaged(table);
        bytes memory dir = abi.encodePacked(tokPtr);
        bytes memory page;
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

        uint32 dim = uint32(spec.readUint(".cfg.dim"));
        uint32 heads = uint32(spec.readUint(".cfg.n_heads"));
        uint32 kv = uint32(spec.readUint(".cfg.n_kv"));
        uint32 hs = dim / heads;
        cfg = Transformer.Config({
            dim: dim,
            hidden: uint32(spec.readUint(".cfg.hidden")),
            nLayers: uint32(spec.readUint(".cfg.n_layers")),
            nHeads: heads,
            nKv: kv,
            vocab: uint32(spec.readUint(".cfg.vocab")),
            seqLen: uint32(spec.readUint(".cfg.seqlen")),
            hs: hs,
            kvd: hs * kv,
            shared: true,
            invSqrtHs: uint64((uint256(1) << 64) / FixedPoint.isqrt(uint256(hs) << 64)),
            nExperts: uint32(spec.readUint(".cfg.n_experts")),
            topK: uint32(spec.readUint(".cfg.top_k"))
        });

        llm = new OnchainLLM(_writePaged(dir), cfg, 0);
    }

    function _writePaged(bytes memory data) internal returns (address) {
        return WeightStore.write(data);
    }

    /// @notice The mixture must generate the specification's tokens, exactly.
    function test_MixtureMatchesSpec() public {
        uint256[] memory promptRaw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");

        uint16[] memory prompt = new uint16[](promptRaw.length);
        for (uint256 i = 0; i < promptRaw.length; ++i) {
            prompt[i] = uint16(promptRaw[i]);
        }

        llm.start(prompt, want.length);
        // Driven to completion rather than for a fixed number of steps, because the phase
        // count a mixture needs depends on how many experts each position visits.
        uint256 guard = 0;
        while (!llm.finished()) {
            llm.step(0);
            require(++guard < 100_000, "generation did not finish");
        }

        uint16[] memory got = llm.generatedIds();
        assertEq(got.length, want.length, "token count differs");
        for (uint256 i = 0; i < want.length; ++i) {
            assertEq(uint256(got[i]), want[i], "generated token diverged from spec");
        }
        console2.log("mixture tokens matched spec:", want.length);
        console2.log("phases run:", llm.steps());
    }
}
