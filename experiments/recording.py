"""Deterministic plotting checkpoints and compact, lossless action blocks."""

import base64
import binascii
import zlib

import numpy as np


MAX_RECORDED_POINTS = 2_000


def recording_checkpoints(horizon: int, max_points: int = MAX_RECORDED_POINTS) -> tuple[int, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if horizon <= max_points:
        return tuple(range(1, horizon + 1))
    # Resolve early transients as well as the long-run curve; never exceed the budget.
    logarithmic = np.geomspace(1, horizon, max_points // 2).astype(int)
    linear = np.linspace(1, horizon, max_points - max_points // 2).astype(int)
    return tuple(sorted({1, horizon, *map(int, logarithmic), *map(int, linear)}))


def encode_action_block(actions: list[int]) -> str:
    payload = np.asarray(actions, dtype="<u4").tobytes()
    encoded = base64.b64encode(zlib.compress(payload)).decode("ascii")
    return f"v1:{len(actions)}:{encoded}"


def action_block_length(block: str) -> int:
    try:
        version, count, _ = block.split(":", 2)
        count = int(count)
        if version != "v1" or count <= 0:
            raise ValueError
        return count
    except (ValueError, AttributeError) as error:
        raise ValueError("invalid action_history block") from error


def decode_action_block(block: str, expected_count: int) -> np.ndarray:
    if action_block_length(block) != expected_count:
        raise ValueError("action_history block does not cover the checkpoint interval")
    try:
        compressed = base64.b64decode(block.split(":", 2)[2], validate=True)
        decoder = zlib.decompressobj()
        payload = decoder.decompress(compressed, expected_count * 4 + 1)
        if len(payload) != expected_count * 4 or not decoder.eof or decoder.unused_data:
            raise ValueError("invalid action_history payload length")
        return np.frombuffer(payload, dtype="<u4")
    except (binascii.Error, zlib.error) as error:
        raise ValueError("invalid action_history payload") from error
