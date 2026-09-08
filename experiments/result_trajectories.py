from collections.abc import Sequence
from itertools import chain
from pathlib import Path

import numpy as np

from experiments.results import iter_result_rows, result_algorithm_profile
from experiments.recording import decode_action_block
from metrics.empirical_distribution import validate_action_shape


def load_result_action_profiles(input_path: str | Path, action_shape: Sequence[int]) -> np.ndarray:
    input_path = Path(input_path)
    shape = validate_action_shape(action_shape)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")
    if len(shape) != len(result_algorithm_profile(first_row)):
        raise ValueError("action shape does not match recorded players")

    if "action_history" in first_row:
        horizon = int(first_row["horizon"])
        profiles = np.empty((horizon, len(shape)), dtype=np.uint32)
        previous_times = [0] * len(shape)
        for row in chain((first_row,), rows):
            player, time = int(row["player"]), int(row["t"])
            start = previous_times[player]
            actions = decode_action_block(row["action_history"], time - start)
            if np.any(actions >= shape[player]) or actions[-1] != int(row["action"]):
                raise ValueError("action_history contains out-of-range or inconsistent actions")
            profiles[start:time, player] = actions
            previous_times[player] = time
        return profiles

    profiles = []
    current_time = 1
    actions: dict[int, int] = {}
    for row in chain((first_row,), rows):
        time = int(row["t"])
        if time != current_time:
            if time != current_time + 1:
                raise ValueError(f"result rounds must be contiguous; expected {current_time + 1}, found {time}")
            profiles.append(tuple(actions[player] for player in range(len(shape))))
            actions = {}
            current_time = time
        player = int(row["player"])
        action = int(row["action"])
        if action < 0 or action >= shape[player]:
            raise ValueError(f"round {time} contains an out-of-range action")
        actions[player] = action

    profiles.append(tuple(actions[player] for player in range(len(shape))))
    return np.asarray(profiles, dtype=int)
