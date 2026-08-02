// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {FixedPoint} from "../src/FixedPoint.sol";

/// @notice The Solidity primitives must agree with the Python specification
///         (tools/intref.py) exactly — not approximately. Any divergence here
///         would silently change generated text.
contract FixedPointTest is Test {
    using stdJson for string;

    string internal vectors;

    function setUp() public {
        vectors = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/fixedpoint_vectors.json"));
    }

    function test_IsqrtMatchesSpec() public view {
        string[][] memory cases = abi.decode(vectors.parseRaw(".isqrt"), (string[][]));
        for (uint256 i = 0; i < cases.length; ++i) {
            uint256 a = vm.parseUint(cases[i][0]);
            uint256 want = vm.parseUint(cases[i][1]);
            assertEq(FixedPoint.isqrt(a), want, "isqrt diverged from spec");
        }
        console2.log("isqrt vectors matched:", cases.length);
    }

    function test_ExpMatchesSpec() public view {
        string[][] memory cases = abi.decode(vectors.parseRaw(".exp"), (string[][]));
        for (uint256 i = 0; i < cases.length; ++i) {
            int256 a = vm.parseInt(cases[i][0]);
            uint256 want = vm.parseUint(cases[i][1]);
            assertEq(FixedPoint.expQ32(a), want, "expQ32 diverged from spec");
        }
        console2.log("exp vectors matched:", cases.length);
    }

    function test_SiluMatchesSpec() public view {
        string[][] memory cases = abi.decode(vectors.parseRaw(".silu"), (string[][]));
        for (uint256 i = 0; i < cases.length; ++i) {
            int256 a = vm.parseInt(cases[i][0]);
            int256 want = vm.parseInt(cases[i][1]);
            assertEq(FixedPoint.silu(a), want, "silu diverged from spec");
        }
        console2.log("silu vectors matched:", cases.length);
    }

    function testFuzz_IsqrtIsFloorSqrt(uint256 a) public pure {
        uint256 r = FixedPoint.isqrt(a);
        assertLe(r, type(uint128).max);
        assertTrue(r * r <= a || r == 0, "not a lower bound");
        if (r < type(uint128).max) assertTrue((r + 1) * (r + 1) > a, "not the floor");
    }
}
