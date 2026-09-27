// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {Transformer} from "../src/Transformer.sol";

/// @notice The expert axis must cost a dense model nothing.
///
/// @dev A mixture's feed-forward tensors are the dense ones with an expert axis, so one
///      expert has to be the dense case rather than a special case of it. If it is not,
///      every blob already deployed — including the two live on Base Sepolia — stops
///      matching the offsets this library derives, silently, and the model reads
///      whichever bytes happen to be there.
///
///      So this pins the property directly rather than trusting that the arithmetic
///      works out: zero experts, one expert, and the field left unset must all produce
///      byte-identical offsets.
contract LayoutTest is Test {
    function _cfg(uint32 nExperts, uint32 topK) internal pure returns (Transformer.Config memory c) {
        // stories15M's geometry, so the numbers below are the ones actually deployed.
        c = Transformer.Config({
            dim: 288,
            hidden: 768,
            nLayers: 6,
            nHeads: 6,
            nKv: 6,
            vocab: 32000,
            seqLen: 256,
            hs: 48,
            kvd: 288,
            shared: true,
            invSqrtHs: 0,
            nExperts: nExperts,
            topK: topK
        });
    }

    function _assertSame(Transformer.Layout memory a, Transformer.Layout memory b) internal pure {
        assertEq(a.tokEmb, b.tokEmb, "tokEmb");
        assertEq(a.rmsAtt, b.rmsAtt, "rmsAtt");
        assertEq(a.wq, b.wq, "wq");
        assertEq(a.wk, b.wk, "wk");
        assertEq(a.wv, b.wv, "wv");
        assertEq(a.wo, b.wo, "wo");
        assertEq(a.rmsFfn, b.rmsFfn, "rmsFfn");
        assertEq(a.w1, b.w1, "w1");
        assertEq(a.w2, b.w2, "w2");
        assertEq(a.w3, b.w3, "w3");
        assertEq(a.rmsFinal, b.rmsFinal, "rmsFinal");
        assertEq(a.wcls, b.wcls, "wcls");
        assertEq(a.ropeCos, b.ropeCos, "ropeCos");
        assertEq(a.ropeSin, b.ropeSin, "ropeSin");
        assertEq(a.router, b.router, "router");
    }

    /// @notice Unset, zero and one expert are the same layout.
    function test_DenseLayoutUnchanged() public pure {
        Transformer.Layout memory unset = Transformer.layout(_cfg(0, 0));
        Transformer.Layout memory one = Transformer.layout(_cfg(1, 1));
        _assertSame(unset, one);
        assertEq(unset.router, 0, "dense must carry no router block");
    }

    /// @notice A mixture grows the feed-forward by exactly the expert count.
    function test_ExpertAxisScalesFeedForward() public pure {
        Transformer.Config memory d = _cfg(1, 1);
        Transformer.Config memory m = _cfg(8, 2);
        Transformer.Layout memory ld = Transformer.layout(d);
        Transformer.Layout memory lm = Transformer.layout(m);

        // Everything before the feed-forward is untouched.
        assertEq(lm.tokEmb, ld.tokEmb, "tokEmb moved");
        assertEq(lm.wq, ld.wq, "wq moved");
        assertEq(lm.rmsFfn, ld.rmsFfn, "rmsFfn moved");

        // The router sits between the norm and the experts, so w1 moves by its size.
        assertGt(lm.router, 0, "mixture must carry a router block");
        assertEq(lm.router, ld.w1, "router should start where the dense w1 did");
        assertEq(lm.w1, lm.router + Transformer.blockSize(6 * 8, d.dim), "w1 after router");

        // Each feed-forward block is eight times the dense one.
        assertEq(lm.w2 - lm.w1, 8 * (ld.w2 - ld.w1) - 0, "w1 block should be 8x");
        assertEq(lm.w3 - lm.w2, 8 * (ld.w3 - ld.w2) - 0, "w2 block should be 8x");
    }

    /// @notice An expert's rows are contiguous, which is what makes an unvisited expert
    ///         bytes never copied rather than bytes skipped over.
    function test_ExpertRowsAreContiguous() public pure {
        Transformer.Config memory c = _cfg(8, 2);
        uint256 h = c.hidden;
        // Layer 0's experts follow one another.
        for (uint256 e = 0; e < 8; ++e) {
            assertEq(Transformer.expertRow(c, 0, e, h), e * h, "layer 0 expert row");
        }
        // Layer 1 starts after all of layer 0's experts.
        assertEq(Transformer.expertRow(c, 1, 0, h), 8 * h, "layer 1 first expert");
        // And a dense config addresses the layer exactly as it always did.
        Transformer.Config memory d = _cfg(0, 0);
        assertEq(Transformer.expertRow(d, 3, 0, h), 3 * h, "dense row addressing changed");
    }
}
