// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {FixedPoint} from "../src/FixedPoint.sol";

/// @notice What a transformer's non-matmul operations cost in the EVM.
///
/// @dev Establishes the second half of the tensor-evm companion project's baseline. The
///      first measured matrix multiplication at 63 gas per multiply-accumulate. This
///      measures everything else: the exponential, SiLU, the square root RMSNorm needs.
///
///      The question is not which is more expensive today — matmul dominates so completely
///      that nothing else registers. It is what happens once matmul is a precompile and
///      stops costing anything. Whatever is left then sets the price, and a design that
///      accelerated only matrix multiplication would be bounded by whatever it did not.
contract OpsCostTest is Test {
    /// Qwen3-0.6B, one layer.
    uint256 constant DIM = 1024;
    uint256 constant HIDDEN = 3072;
    uint256 constant HEADS = 16;
    uint256 constant KV = 8;
    uint256 constant HS = 128;
    /// dim*(q + k + v + o) + 3 * hidden * dim, at this geometry.
    uint256 constant MACS_PER_LAYER = 15_728_640;
    /// Measured in test/MatmulBench.t.sol.
    uint256 constant GAS_PER_MAC = 63;
    /// tensor-evm prices the precompile at one gas per 1024 multiply-accumulates.
    uint256 constant MACS_PER_GAS_PRECOMPILE = 1024;

    function _exp(int256 x) internal pure returns (uint256) {
        return FixedPoint.expQ32(x);
    }

    function test_OpsCost() public view {
        int256 one = FixedPoint.ONE;

        // ---- exponential ----
        uint256 g = gasleft();
        for (uint256 i = 0; i < 16; ++i) {
            _exp(-int256(i + 1) * one / 4);
        }
        uint256 expGas = (g - gasleft()) / 16;

        // ---- SiLU, both branches, since the negative one takes a different path ----
        g = gasleft();
        for (uint256 i = 0; i < 16; ++i) {
            FixedPoint.silu(int256(i) * one / 4 - 2 * one);
        }
        uint256 siluGas = (g - gasleft()) / 16;

        // ---- square root, as RMSNorm calls it once per vector ----
        g = gasleft();
        for (uint256 i = 1; i <= 16; ++i) {
            FixedPoint.isqrt(i * 1e18);
        }
        uint256 sqrtGas = (g - gasleft()) / 16;

        console2.log("exp   (per call) :", expGas);
        console2.log("silu  (per call) :", siluGas);
        console2.log("isqrt (per call) :", sqrtGas);

        // ---- what a layer costs, before and after a matmul precompile ----
        uint256 matmulEvm = MACS_PER_LAYER * GAS_PER_MAC;
        uint256 matmulPrecompile = MACS_PER_LAYER / MACS_PER_GAS_PRECOMPILE;
        // SwiGLU runs SiLU once per hidden unit. Softmax runs exp once per attended
        // position; at short context that is small, so SwiGLU is the term that matters.
        uint256 swiglu = HIDDEN * siluGas;
        // RMSNorm: twice on the residual, once per query and key head for QK-norm.
        uint256 norms = (2 + HEADS + KV) * sqrtGas;

        console2.log("");
        console2.log("one Qwen3-0.6B layer, gas");
        console2.log("  matmul in the EVM      :", matmulEvm);
        console2.log("  matmul via precompile  :", matmulPrecompile);
        console2.log("  swiglu (3072 x silu)   :", swiglu);
        console2.log("  rmsnorm (26 x isqrt)   :", norms);

        console2.log("");
        console2.log("share of a layer spent outside matmul");
        console2.log("  today, percent           :", (100 * (swiglu + norms)) / (matmulEvm + swiglu + norms));
        console2.log("  with a matmul precompile :", (100 * (swiglu + norms)) / (matmulPrecompile + swiglu + norms));
        console2.log("  swiglu / matmul, after   :", swiglu / matmulPrecompile);
    }
}
