"""Deterministic checkpoints and cumulative joint-action histograms."""

import json

import numpy as np


MAX_RECORDED_POINTS = 500
MAX_EQUILIBRIUM_POINTS = 20


def recording_checkpoints(horizon: int, max_points: int = MAX_RECORDED_POINTS) -> tuple[int, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    max_points = min(max_points, MAX_RECORDED_POINTS)
    if horizon <= max_points:
        return tuple(range(1, horizon + 1))
    logarithmic = np.geomspace(1, horizon, max_points).astype(int)
    logarithmic[0], logarithmic[-1] = 1, horizon
    return tuple(sorted(set(map(int, logarithmic))))


def joint_action_histogram_checkpoints(horizon: int) -> tuple[int, ...]:
    return recording_checkpoints(horizon, MAX_EQUILIBRIUM_POINTS)


def encode_joint_action_histograms(horizons: list[int], counts: list[np.ndarray]) -> str:
    return json.dumps(
        {"horizons": horizons, "counts": [values.ravel(order="C").tolist() for values in counts]},
        separators=(",", ":"),
    )
