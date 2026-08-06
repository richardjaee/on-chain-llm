// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {Transformer} from "../src/Transformer.sol";

/// @notice Gas cost of the matmul, as the baseline a native precompile is measured against.
///
///         `Transformer.matmul` is already the optimised version: inline assembly, one
///         mload unpacking sixteen int16 weights, the inner loop fully unrolled. This is
///         not a strawman — it is as fast as the operation gets inside the EVM, so
///         whatever gap remains is the interpreter itself rather than the implementation.
///
///         Gas rather than wall clock, deliberately. Forge executes on revm, which is far
///         faster than a node under consensus, so timing it would flatter the EVM. Gas is
///         what a chain actually charges and what decides whether the work fits in a
///         block, which is the quantity that matters.
contract MatmulBenchTest is Test {
    /// @dev Tensor block laid out the way Transformer expects: `rowsTotal` shift bytes,
    ///      then row-major int16 values.
    function _tensor(uint256 rows, uint256 cols) internal pure returns (bytes memory w) {
        w = new bytes(rows + rows * cols * 2);
        for (uint256 r = 0; r < rows; ++r) {
            w[r] = bytes1(uint8(30));
        }
        uint256 off = rows;
        for (uint256 i = 0; i < rows * cols; ++i) {
            int16 v = int16(int256((i * 2654435761) % 4001) - 2000);
            w[off + i * 2] = bytes1(uint8(uint16(v) >> 8));
            w[off + i * 2 + 1] = bytes1(uint8(uint16(v)));
        }
    }

    function _run(uint256 rows, uint256 cols) internal view returns (uint256 gas) {
        bytes memory w = _tensor(rows, cols);
        int256[] memory x = new int256[](cols);
        for (uint256 i = 0; i < cols; ++i) {
            x[i] = int256((i * 40503 + 17) % 131071) - 65535;
        }
        int256[] memory out = new int256[](rows);

        uint256 g = gasleft();
        Transformer.matmul(w, 0, rows, 0, rows, cols, x, out, 0);
        gas = g - gasleft();
    }

    function test_MatmulGas() public view {
        uint256[2][3] memory shapes = [[uint256(576), 576], [uint256(1024), 1024], [uint256(2048), 2048]];

        for (uint256 i = 0; i < shapes.length; ++i) {
            uint256 rows = shapes[i][0];
            uint256 cols = shapes[i][1];
            uint256 gas = _run(rows, cols);
            uint256 macs = rows * cols;

            console2.log("shape rows x cols :", rows, cols);
            console2.log("  weight bytes    :", macs * 2);
            console2.log("  gas             :", gas);
            console2.log("  gas per MAC     :", gas / macs);
            // Ethereum sustains roughly 3M gas/sec (36M gas blocks every 12s), so gas
            // converts directly into how much chain time the work would occupy.
            console2.log("  ms of chain time:", (gas * 1000) / 3_000_000);
        }
    }
}
