from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path

import numpy as np

from experiments.results import iter_result_rows
from metrics.empirical_distribution import EmpiricalDistributionTrajectory, empirical_distribution_trajectory, validate_action_shape


def load_result_action_profiles(input_path: str | Path, action_shape: Sequence[int]) -> np.ndarray:
    input_path = Path(input_path)
    shape = validate_action_shape(action_shape)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")

    profiles = []
    current_time = 1
    actions: dict[int, int] = {}
    for row in chain((first_row,), rows):
        time = int(row["t"])
        if time != current_time:
            if time != current_time + 1:
                raise ValueError(f"result rounds must be contiguous; expected {current_time + 1}, found {time}")
            if set(actions) != set(range(len(shape))):
                raise ValueError(f"round {current_time} has incomplete player actions")
            profiles.append(tuple(actions[player] for player in range(len(shape))))
            actions = {}
            current_time = time
        player = int(row["player"])
        if not 0 <= player < len(shape) or player in actions:
            raise ValueError(f"round {time} contains an invalid or duplicate player")
        action = int(row["action"])
        if action < 0 or action >= shape[player]:
            raise ValueError(f"round {time} contains an out-of-range action")
        actions[player] = action

    if set(actions) != set(range(len(shape))):
        raise ValueError(f"round {current_time} has incomplete player actions")
    profiles.append(tuple(actions[player] for player in range(len(shape))))
    return np.asarray(profiles, dtype=int)


def load_empirical_distribution_trajectory(input_path: str | Path, action_shape: Sequence[int],
                                           checkpoints: Iterable[int] | None = None) -> EmpiricalDistributionTrajectory:
    profiles = load_result_action_profiles(input_path, action_shape)
    return empirical_distribution_trajectory(profiles, action_shape, checkpoints)
