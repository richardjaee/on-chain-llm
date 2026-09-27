// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {Mixture} from "../src/Mixture.sol";
import {FixedPoint} from "../src/FixedPoint.sol";

/// @notice Routing must agree with the specification on every case, including the ties.
///
/// @dev The vectors come from `tools/moevec.rs` in the companion project, which is the
///      implementation that already runs a real 30B mixture, and `tools/moeref.py` is
///      checked against the same file. Three implementations, one set of numbers.
///
///      The cases are chosen for their ties rather than their coverage: every score
///      identical, a tie at the top, scores far enough apart that every exponent
///      underflows. Random scores exercise none of those and would pass a comparator that
///      forks the chain in production.
contract MixtureTest is Test {
    using stdJson for string;
    struct Vec {
        uint256 k;
        int256[] scores;
        uint256[] idx;
        int256[] val;
        int256[] w;
        uint256[] order;
    }

    /// @dev Signed values are quoted in the file and parsed back, the way the existing
    ///      vectors here do it: `stdJson` reads unsigned arrays directly but has no
    ///      signed equivalent.
    function _ints(string memory raw, string memory path) internal pure returns (int256[] memory out) {
        string[] memory s = abi.decode(raw.parseRaw(path), (string[]));
        out = new int256[](s.length);
        for (uint256 i = 0; i < s.length; ++i) {
            out[i] = vm.parseInt(s[i]);
        }
    }

    function _load() internal view returns (Vec[] memory out) {
        string memory raw = vm.readFile(string.concat(vm.projectRoot(), "/artifacts/moe_vectors.json"));
        uint256 n = raw.readUint(".n");
        out = new Vec[](n);
        for (uint256 i = 0; i < n; ++i) {
            string memory p = string.concat(".c", vm.toString(i));
            out[i].k = raw.readUint(string.concat(p, ".k"));
            out[i].scores = _ints(raw, string.concat(p, ".scores"));
            out[i].idx = raw.readUintArray(string.concat(p, ".idx"));
            out[i].val = _ints(raw, string.concat(p, ".val"));
            out[i].w = _ints(raw, string.concat(p, ".w"));
            out[i].order = raw.readUintArray(string.concat(p, ".order"));
        }
    }

    function test_RoutingMatchesSpec() public view {
        Vec[] memory vs = _load();
        assertGt(vs.length, 0, "no vectors");

        for (uint256 c = 0; c < vs.length; ++c) {
            Vec memory t = vs[c];

            (uint256[] memory idx, int256[] memory val) = Mixture.topK(t.scores, t.k);
            assertEq(idx.length, t.idx.length, "chosen count");
            for (uint256 i = 0; i < idx.length; ++i) {
                assertEq(idx[i], t.idx[i], "expert index");
                assertEq(val[i], t.val[i], "expert score");
            }

            int256[] memory w = Mixture.routeWeights(val);
            assertEq(w.length, t.w.length, "weight count");
            for (uint256 i = 0; i < w.length; ++i) {
                assertEq(w[i], t.w[i], "route weight");
            }

            uint256[] memory order = Mixture.visitOrder(idx);
            assertEq(order.length, t.order.length, "order count");
            for (uint256 i = 0; i < order.length; ++i) {
                assertEq(order[i], t.order[i], "visit order");
            }
        }
    }

    /// @dev The weights are a distribution, so they sum to one — but in Q32 with a floored
    ///      division, so the sum is one minus at most k ulps. Asserting the bound rather
    ///      than equality states what the arithmetic actually promises.
    function test_WeightsSumToOne() public view {
        Vec[] memory vs = _load();
        for (uint256 c = 0; c < vs.length; ++c) {
            (, int256[] memory val) = Mixture.topK(vs[c].scores, vs[c].k);
            int256[] memory w = Mixture.routeWeights(val);
            int256 sum = 0;
            for (uint256 i = 0; i < w.length; ++i) {
                assertGe(w[i], 0, "weight sign");
                sum += w[i];
            }
            assertLe(sum, FixedPoint.ONE, "weights exceed one");
            assertGe(sum, FixedPoint.ONE - int256(w.length), "weights lost more than k ulps");
        }
    }

    /// @dev The rule that keeps a chain from forking, stated on its own so a regression
    ///      names itself. Every score equal means every slot must be the lowest index not
    ///      already taken.
    function test_TiesGoToLowestIndex() public pure {
        int256[] memory flat = new int256[](6);
        (uint256[] memory idx,) = Mixture.topK(flat, 4);
        for (uint256 i = 0; i < idx.length; ++i) {
            assertEq(idx[i], i, "tie broken away from lowest index");
        }
    }
}
