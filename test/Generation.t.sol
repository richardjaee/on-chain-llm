// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {FixedPoint} from "../src/FixedPoint.sol";
import {Transformer} from "../src/Transformer.sol";

/// @notice End-to-end: the Solidity forward pass must generate exactly the token
///         sequence our Python specification produces on the real checkpoint.
contract GenerationTest is Test {
    using stdJson for string;

    bytes internal blob;
    Transformer.Config internal cfg;
    Transformer.Layout internal lay;

    function setUp() public {
        string memory root = vm.projectRoot();
        blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));
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
            invSqrtHs: uint64((uint256(1) << 64) / FixedPoint.isqrt(uint256(8) << 64))
        });
        lay = Transformer.layout(cfg);
    }

    function _generate(uint16[] memory prompt, uint256 maxNew) internal view returns (uint256[] memory gen) {
        uint256 maxPos = prompt.length + maxNew;
        Transformer.Buffers memory b = Transformer.newBuffers(cfg, maxPos);
        gen = new uint256[](maxNew);
        uint256 n = 0;
        uint256 token = prompt[0];
        for (uint256 pos = 0; pos + 1 < maxPos; ++pos) {
            Transformer.forwardRange(blob, lay, cfg, b, token, pos, 0, cfg.nLayers);
            Transformer.normFinal(blob, lay, cfg, b);
            Transformer.classifyRange(blob, lay, cfg, b, 0, cfg.vocab);
            uint256 next;
            if (pos + 1 < prompt.length) {
                next = prompt[pos + 1];
            } else {
                next = FixedPoint.argmax(b.logits);
                gen[n++] = next;
                if (next == 1 || next == 2) break;
            }
            token = next;
        }
        assembly {
            mstore(gen, n)
        }
    }

    /// @notice argmax after the first prompt token must match the spec
    function test_FirstLogitsMatchSpec() public view {
        string memory spec = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/spec_vectors.json"));
        uint256 wantArg = spec.readUint(".argmax0");

        Transformer.Buffers memory b = Transformer.newBuffers(cfg, 4);
        Transformer.forwardRange(blob, lay, cfg, b, 403, 0, 0, cfg.nLayers);
        Transformer.normFinal(blob, lay, cfg, b);
        Transformer.classifyRange(blob, lay, cfg, b, 0, cfg.vocab);

        string[] memory wantLogits = abi.decode(spec.parseRaw(".logits0"), (string[]));
        for (uint256 i = 0; i < wantLogits.length; ++i) {
            assertEq(b.logits[i], vm.parseInt(wantLogits[i]), "logit diverged from spec");
        }
        assertEq(FixedPoint.argmax(b.logits), wantArg, "argmax diverged from spec");
        console2.log("logits matched spec:", wantLogits.length);
        console2.log("argmax:", wantArg);
    }

    /// @notice Full generation must match the spec's token sequence exactly
    function test_GenerationMatchesSpec() public view {
        string memory spec = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/generation_vectors.json"));
        uint256[] memory promptRaw = spec.readUintArray(".prompt");
        uint256[] memory want = spec.readUintArray(".gen");

        uint16[] memory prompt = new uint16[](promptRaw.length);
        for (uint256 i = 0; i < promptRaw.length; ++i) {
            prompt[i] = uint16(promptRaw[i]);
        }

        uint256[] memory got = _generate(prompt, want.length);
        assertEq(got.length, want.length, "token count differs");
        for (uint256 i = 0; i < want.length; ++i) {
            assertEq(got[i], want[i], "generated token diverged from spec");
        }
        console2.log("generated tokens matching spec:", want.length);
    }
}
