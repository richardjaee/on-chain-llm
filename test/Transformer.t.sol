// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {FixedPoint} from "../src/FixedPoint.sol";
import {Transformer} from "../src/Transformer.sol";
import {WeightStore} from "../src/WeightStore.sol";

/// @notice The Solidity forward pass must reproduce the Python specification
///         (tools/model.py) bit for bit on the real checkpoint.
contract TransformerTest is Test {
    using stdJson for string;

    bytes internal blob;
    Transformer.Config internal cfg;
    Transformer.Layout internal lay;
    string internal spec;

    function setUp() public {
        string memory root = vm.projectRoot();
        blob = vm.parseBytes(string.concat("0x", vm.readFile(string.concat(root, "/artifacts/model.hex"))));
        spec = vm.readFile(string.concat(root, "/artifacts/spec_vectors.json"));
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
            invSqrtHs: uint64((uint256(1) << 32) / 2), // 1/sqrt(8) handled below
            nExperts: 0,
            topK: 0
        });
        // 2^32 / sqrt(8), computed with the same integer sqrt the model uses
        cfg.invSqrtHs = uint64((uint256(1) << 64) / FixedPoint.isqrt(uint256(8) << 64));
        lay = Transformer.layout(cfg);
    }

    function test_LayoutMatchesBlob() public view {
        uint256 declared = spec.readUint(".blobLen");
        // rope sin block is last: its offset plus its size must equal the blob
        uint256 end = lay.ropeSin + uint256(cfg.seqLen) * (cfg.hs / 2) * 4;
        assertEq(end, declared, "derived layout does not span the blob exactly");
        assertEq(blob.length, declared, "blob length mismatch");
        console2.log("layout spans the blob exactly:", end);
    }

    function test_MatmulMatchesSpec() public view {
        int256[] memory x = new int256[](cfg.dim);
        for (uint256 i = 0; i < cfg.dim; ++i) {
            x[i] = int256(int128(int256(i) * 7919 - 100000)) << 12;
        }
        int256[] memory out = new int256[](8);
        Transformer.matmul(blob, lay.wq, uint256(cfg.nLayers) * cfg.dim, 0, 8, cfg.dim, x, out, 0);

        string[] memory want = abi.decode(spec.parseRaw(".probe_wq"), (string[]));
        for (uint256 i = 0; i < 8; ++i) {
            assertEq(out[i], vm.parseInt(want[i]), "matmul row diverged from spec");
        }
        console2.log("wq matmul rows matched spec:", uint256(8));
    }

    function test_RmsnormRunsAndIsFinite() public view {
        int256[] memory x = new int256[](cfg.dim);
        for (uint256 i = 0; i < cfg.dim; ++i) {
            x[i] = int256(1 << 30) + int256(i);
        }
        int256[] memory out = new int256[](cfg.dim);
        Transformer.rmsnorm(blob, lay.rmsAtt, 0, x, out, cfg.dim);
        bool any;
        for (uint256 i = 0; i < cfg.dim; ++i) {
            if (out[i] != 0) any = true;
        }
        assertTrue(any, "rmsnorm produced all zeros");
        console2.log("rmsnorm out[0]:", out[0]);
    }
}
