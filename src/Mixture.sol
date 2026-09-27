// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {FixedPoint} from "./FixedPoint.sol";

/// @notice Routing for a mixture-of-experts feed-forward: which experts a token visits,
///         and how their outputs are weighted.
///
/// @dev Gas follows the parameters a token *reads*, and a dense layer reads all of them.
///      A router that picks 8 experts of 128 skips most of the layer, and every skipped
///      chunk is a data contract never touched — which is why a mixture is the one lever
///      that moves this project's ceiling by an order of magnitude rather than a few
///      percent. Storage is billed once; reading is billed again by every node, forever.
///
///      Two things must be reproducible, and the second is the one that bites. The
///      arithmetic, as everywhere here. And **which experts are chosen**: a tie broken
///      differently is not a rounding difference, it sends the token through different
///      parameters, so two honest nodes emit different text and the chain forks. Ties go
///      to the lower index, everywhere, always.
///
///      Matches `tools/moeref.py`, which matches the companion project's `moe.rs`.
library Mixture {
    /// @notice The `k` largest of `v`, as parallel arrays, highest first.
    /// @dev A selection sort rather than anything cleverer. `k` is 8 against at most a few
    ///      hundred experts, so the quadratic term is small, and the tie rule is the point:
    ///      a candidate replaces the incumbent only when *strictly* greater, which leaves
    ///      the first occurrence of a repeated maximum holding the slot. A comparator that
    ///      merely looks right passes on random scores and forks the chain on equal ones.
    function topK(int256[] memory v, uint256 k)
        internal
        pure
        returns (uint256[] memory idx, int256[] memory val)
    {
        uint256 n = v.length;
        if (k > n) k = n;
        idx = new uint256[](k);
        val = new int256[](k);

        bool[] memory taken = new bool[](n);
        for (uint256 slot = 0; slot < k; ++slot) {
            bool found = false;
            uint256 bi = 0;
            int256 bv = 0;
            for (uint256 i = 0; i < n; ++i) {
                if (taken[i]) continue;
                if (!found || v[i] > bv) {
                    found = true;
                    bi = i;
                    bv = v[i];
                }
            }
            if (!found) break;
            taken[bi] = true;
            idx[slot] = bi;
            val[slot] = bv;
        }
    }

    /// @notice Softmax over the chosen scores alone, in Q32.
    /// @dev Over the top `k` rather than over all experts, which is what Qwen3 and DeepSeek
    ///      do: the weights of the experts a token actually visits are renormalised among
    ///      themselves. Normalising over everything and then dropping the unchosen ones
    ///      would leave the kept weights summing to less than one, quietly scaling the
    ///      whole feed-forward output down.
    function routeWeights(int256[] memory chosenVal) internal pure returns (int256[] memory w) {
        uint256 k = chosenVal.length;
        w = new int256[](k);
        if (k == 0) return w;

        // Shift by the maximum before exponentiating. Scores are logits and can be large
        // and positive; expQ32 of a large positive number overflows, and subtracting the
        // maximum is exact in fixed point, so the shift costs nothing.
        int256 mx = chosenVal[0];
        for (uint256 i = 1; i < k; ++i) {
            if (chosenVal[i] > mx) mx = chosenVal[i];
        }

        uint256 total = 0;
        for (uint256 i = 0; i < k; ++i) {
            uint256 e = FixedPoint.expQ32(chosenVal[i] - mx);
            w[i] = int256(e);
            total += e;
        }

        if (total == 0) {
            // Every exponent underflowed, so the scores are spread too far apart for the
            // arithmetic to separate them. Falling back to the first expert is
            // deterministic; dividing is not.
            for (uint256 i = 0; i < k; ++i) {
                w[i] = 0;
            }
            w[0] = FixedPoint.ONE;
            return w;
        }

        for (uint256 i = 0; i < k; ++i) {
            w[i] = int256((uint256(w[i]) << FixedPoint.F) / total);
        }
    }

    /// @notice Slots of the chosen set, reordered so experts are visited by index.
    /// @dev The sum over experts is integer and therefore associative, so the order cannot
    ///      change the result. It is fixed anyway: a reader has no way to know the order is
    ///      irrelevant without checking it, and one fewer thing to prove about consensus is
    ///      worth an insertion sort over at most `k` elements.
    function visitOrder(uint256[] memory idx) internal pure returns (uint256[] memory order) {
        uint256 k = idx.length;
        order = new uint256[](k);
        for (uint256 i = 0; i < k; ++i) {
            order[i] = i;
        }
        for (uint256 i = 1; i < k; ++i) {
            uint256 cur = order[i];
            uint256 j = i;
            while (j > 0 && idx[order[j - 1]] > idx[cur]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = cur;
        }
    }

    /// @notice SiLU(gate) * up, elementwise, Q32 in and out.
    function swiglu(int256[] memory gate, int256[] memory up, int256[] memory out) internal pure {
        for (uint256 i = 0; i < gate.length; ++i) {
            out[i] = (FixedPoint.silu(gate[i]) * up[i]) >> FixedPoint.F;
        }
    }
}
