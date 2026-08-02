// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

/// @title WeightStore
/// @notice Stores immutable byte blobs as contract runtime code and reads them back
///         with EXTCODECOPY, including arbitrary sub-ranges spanning several chunks.
/// @dev The "data contract" idea (deploy bytes as code, read with EXTCODECOPY) is a
///      well-known public technique. A leading STOP byte guarantees the payload can
///      never execute as meaningful code; reads skip it.
///
///      `readRange` is what makes large models viable: EVM memory expansion is
///      quadratic (3w + w^2/512), so materialising a whole blob passes a 14M gas
///      budget at roughly 2.7MB regardless of how finely the compute is sharded.
///      Reading only the rows a shard needs makes peak memory a constant the caller
///      chooses rather than a function of model size.
library WeightStore {
    /// @notice Max payload per chunk: EIP-170's 24,576 limit minus the STOP prefix
    uint256 internal constant MAX_PAYLOAD = 24_575;

    error PayloadTooLarge();
    error DeployFailed();
    error OutOfRange();

    /// @notice Deploy `payload` as the runtime code of a new data contract
    function write(bytes memory payload) internal returns (address ptr) {
        if (payload.length > MAX_PAYLOAD) revert PayloadTooLarge();
        // Ten-byte constructor, using RETURNDATASIZE as a one-byte zero (it is
        // guaranteed 0 during construction, and costs 2 gas against PUSH1 0x00's 3):
        //   61 LLLL  PUSH2 runtimeLen
        //   80       DUP1
        //   600A     PUSH1 10          offset of the runtime within this initcode
        //   3D       RETURNDATASIZE    -> 0, the destination in memory
        //   39       CODECOPY
        //   3D       RETURNDATASIZE    -> 0, the return offset
        //   F3       RETURN
        // The runtime that follows is `0x00 || payload`; the leading STOP means the
        // blob can never execute as meaningful code, and reads skip that byte.
        bytes memory initCode =
            abi.encodePacked(hex"61", uint16(payload.length + 1), hex"80600A3D393DF3", hex"00", payload);
        assembly ("memory-safe") {
            ptr := create(0, add(initCode, 0x20), mload(initCode))
        }
        if (ptr == address(0)) revert DeployFailed();
    }

    /// @notice Payload length of a data contract
    function size(address ptr) internal view returns (uint256 n) {
        assembly ("memory-safe") {
            n := extcodesize(ptr)
        }
        if (n == 0) revert DeployFailed();
        unchecked {
            n -= 1;
        }
    }

    /// @notice Copy the whole payload of `ptr`
    function read(address ptr) internal view returns (bytes memory out) {
        uint256 n = size(ptr);
        out = new bytes(n);
        assembly ("memory-safe") {
            extcodecopy(ptr, add(out, 0x20), 1, n)
        }
    }

    /// @notice Copy bytes [offset, offset+len) of the LOGICAL blob into `dest`
    /// @dev The logical blob is every chunk's payload concatenated, each exactly
    ///      MAX_PAYLOAD except the last. A range may straddle chunks, so this walks
    ///      them and issues one EXTCODECOPY per overlap.
    function readRange(address[] memory chunks, uint256 offset, uint256 len, bytes memory dest) internal view {
        if (len > dest.length) revert OutOfRange();
        uint256 done = 0;
        while (done < len) {
            uint256 pos = offset + done;
            uint256 ci = pos / MAX_PAYLOAD;
            if (ci >= chunks.length) revert OutOfRange();
            uint256 within = pos % MAX_PAYLOAD;
            uint256 take = MAX_PAYLOAD - within;
            uint256 left = len - done;
            if (take > left) take = left;
            address chunk = chunks[ci];
            assembly ("memory-safe") {
                extcodecopy(chunk, add(add(dest, 0x20), done), add(1, within), take)
            }
            unchecked {
                done += take;
            }
        }
    }

    /// @notice Read a single byte of the logical blob (used for per-row shifts)
    function readByte(address[] memory chunks, uint256 offset) internal view returns (uint8 b) {
        uint256 ci = offset / MAX_PAYLOAD;
        if (ci >= chunks.length) revert OutOfRange();
        uint256 within = offset % MAX_PAYLOAD;
        address chunk = chunks[ci];
        bytes memory one = new bytes(32);
        assembly ("memory-safe") {
            extcodecopy(chunk, add(one, 0x20), add(1, within), 1)
            b := byte(0, mload(add(one, 0x20)))
        }
    }
}
