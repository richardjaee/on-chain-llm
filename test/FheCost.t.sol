// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {Test, console2} from "forge-std/Test.sol";

/// @notice What would it cost to run the model on ENCRYPTED data in the EVM?
///
///         Keeping generation on chain means validators re-execute it, which means
///         every value they touch is public. The only escape is for the values to be
///         ciphertexts — homomorphic encryption — so the EVM computes without ever
///         seeing plaintext. That is the one construction that gives private outputs
///         without moving the computation off chain.
///
///         Lattice FHE (BFV/BGV/CKKS) spends nearly all its time in the number
///         theoretic transform: a butterfly of one modular multiply and two modular
///         adds, n/2 * log2(n) times per transform. This measures that butterfly in
///         the EVM and extrapolates, so the comparison below is grounded in what the
///         machine actually charges rather than in an estimate.
contract FheCostTest is Test {
    /// @dev One NTT pass over `n` coefficients, in the shape a real implementation uses
    function _ntt(uint256 n, uint256 q, uint256 w) internal pure returns (uint256 acc) {
        unchecked {
            acc = 1;
            uint256 stages = 0;
            for (uint256 t = n; t > 1; t >>= 1) stages++;
            for (uint256 s = 0; s < stages; ++s) {
                for (uint256 i = 0; i < n / 2; ++i) {
                    uint256 u = acc;
                    uint256 v = mulmod(acc, w, q);
                    acc = addmod(u, v, q);
                    uint256 d = addmod(u, q - v, q);
                    acc = addmod(acc, d, q) >> 1;
                }
            }
        }
    }

    function test_FheProjection() public view {
        // BFV at a security level that supports meaningful multiplicative depth
        uint256 n = 8192;
        uint256 q = 0xFFFFFFFFFFFFFFC5; // a 64-bit prime-ish modulus
        uint256 w = 0x123456789ABCDEF;

        uint256 g = gasleft();
        _ntt(n, q, w);
        uint256 nttGas = g - gasleft();
        console2.log("one NTT (n=8192)            :", nttGas);

        // A ciphertext-ciphertext multiply is several forward/inverse transforms plus
        // relinearisation, which is itself key-switching over more transforms. Ten is
        // a generous floor; real implementations use more.
        uint256 ctMul = nttGas * 10;
        console2.log("one ciphertext multiply     :", ctMul);

        // stories260K: dim 64, 5 layers, vocab 512 -> about 520,000 multiply-accumulates
        // per token. Under SIMD packing a ciphertext holds n/2 = 4096 slots, so the
        // linear algebra needs far fewer ciphertext ops than scalar MACs. Assume the
        // best case of perfect packing, ignore rotations and key-switching entirely,
        // and ignore that softmax and SiLU are not polynomial at all.
        uint256 macs = 520_000;
        uint256 ctOps = macs / (n / 2);
        if (ctOps == 0) ctOps = 1;
        // depth: 5 layers x roughly 6 multiplicative levels, each needing a bootstrap
        // in practice; count only the multiplies, not the bootstraps
        ctOps = ctOps * 30;
        uint256 perToken = ctOps * ctMul;

        console2.log("ciphertext ops per token    :", ctOps);
        console2.log("FHE gas per token (floor)   :", perToken);
        console2.log("plaintext gas per token     :", uint256(31_000_000));
        console2.log("ratio                       :", perToken / 31_000_000);
    }
}
