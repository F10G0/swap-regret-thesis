import json
from collections.abc import Sequence
from itertools import chain
from math import prod
from pathlib import Path

import numpy as np

from experiments.recording import joint_action_histogram_checkpoints
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD
from experiments.results import iter_result_rows, result_algorithm_profile
from metrics.empirical_distribution import EmpiricalDistributionTrajectory, validate_action_shape


def load_result_empirical_distribution_trajectory(input_path: str | Path, action_shape: Sequence[int]) -> EmpiricalDistributionTrajectory:
    input_path = Path(input_path)
    shape = validate_action_shape(action_shape)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")
    if len(shape) != len(result_algorithm_profile(first_row)):
        raise ValueError("action shape does not match recorded players")
    if JOINT_ACTION_HISTOGRAM_FIELD not in first_row:
        raise ValueError(f"result is missing {JOINT_ACTION_HISTOGRAM_FIELD}")

    horizon = int(first_row["horizon"])
    serialized_payload = None
    for row in chain((first_row,), rows):
        player = int(row["player"])
        action = int(row["action"])
        if action < 0 or action >= shape[player]:
            raise ValueError(f"round {row['t']} contains an out-of-range action")
        serialized = row[JOINT_ACTION_HISTOGRAM_FIELD].strip()
        if serialized:
            serialized_payload = serialized
    assert serialized_payload is not None

    try:
        payload = json.loads(serialized_payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid joint-action histogram JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"horizons", "counts"}:
        raise ValueError("joint-action histogram payload must contain horizons and counts")
    horizons = payload["horizons"]
    counts = payload["counts"]
    expected_horizons = joint_action_histogram_checkpoints(horizon)
    if (
        not isinstance(horizons, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in horizons)
        or tuple(horizons) != expected_horizons
    ):
        raise ValueError("joint-action histogram horizons do not follow the checkpoint policy")
    if not isinstance(counts, list) or len(counts) != len(horizons):
        raise ValueError("joint-action histogram counts do not match the stored horizons")

    profile_count = prod(shape)
    validated_counts = []
    for checkpoint, values in zip(horizons, counts):
        if (
            not isinstance(values, list)
            or len(values) != profile_count
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
        ):
            raise ValueError("joint-action histogram counts must be nonnegative integer vectors of the action-space size")
        if sum(values) != checkpoint:
            raise ValueError("joint-action histogram counts must sum to their horizon")
        validated_counts.append(values)

    try:
        count_matrix = np.asarray(validated_counts, dtype=np.int64)
        stored_horizons = np.asarray(horizons, dtype=int)
    except (OverflowError, ValueError) as error:
        raise ValueError("joint-action histogram values exceed the supported integer range") from error
    if len(count_matrix) > 1 and np.any(np.diff(count_matrix, axis=0) < 0):
        raise ValueError("joint-action histogram counts must be cumulative")
    vectors = count_matrix / np.asarray(horizons, dtype=float)[:, None]
    return EmpiricalDistributionTrajectory(shape, stored_horizons, vectors)
