// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {WeightStore} from "./WeightStore.sol";

/// @title Decoder
/// @notice Turns generated token ids back into UTF-8 using an on-chain token table.
/// @dev Only decoding lives on chain. Prompts arrive already tokenised, because a
///      greedy BPE encoder costs O(len^2 * vocab) in gas — the merge loop rescans the
///      whole vocabulary for every adjacent pair — while decoding is a table lookup.
///
///      Table layout (ours): [u16 count][u8 maxLen][u32 offset per token][bytes].
///      Offsets are 32-bit because a 32k-token vocabulary needs ~178KB of strings, and
///      a 16-bit offset caps at 65,535 — it addressed stories260K's 512 tokens fine and
///      would have silently decoded any real vocabulary to nonsense.
///      Byte-fallback pieces spelled "<0xNN>" map back to the raw byte, matching
///      llama2.c's decode (MIT).
library Decoder {
    uint256 internal constant BOS = 1;
    uint256 internal constant EOS = 2;

    /// @dev Big-endian u32 offset for token `id`
    function _off(bytes memory table, uint256 offBase, uint256 id) private pure returns (uint256) {
        uint256 k = offBase + id * 4;
        return uint256(uint8(table[k])) << 24 | uint256(uint8(table[k + 1])) << 16 | uint256(uint8(table[k + 2])) << 8
            | uint256(uint8(table[k + 3]));
    }

    function decode(bytes memory table, uint16[] memory ids, uint256 count, uint256 prev)
        internal
        pure
        returns (bytes memory out)
    {
        uint256 n = uint256(uint8(table[0])) << 8 | uint256(uint8(table[1]));
        uint256 offBase = 3;
        uint256 strBase = offBase + n * 4;
        out = "";
        for (uint256 i = 0; i < count; ++i) {
            uint256 id = ids[i];
            uint256 s = _off(table, offBase, id);
            uint256 e = id + 1 < n ? _off(table, offBase, id + 1) : table.length - strBase;
            bytes memory piece = new bytes(e - s);
            for (uint256 j = 0; j < e - s; ++j) {
                piece[j] = table[strBase + s + j];
            }
            // the piece after BOS drops a leading space
            if (prev == BOS && piece.length > 0 && piece[0] == 0x20) {
                bytes memory t = new bytes(piece.length - 1);
                for (uint256 j = 1; j < piece.length; ++j) {
                    t[j - 1] = piece[j];
                }
                piece = t;
            }
            out = bytes.concat(out, piece);
            prev = id;
        }
    }
}
