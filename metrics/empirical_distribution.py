from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from operator import index

import numpy as np


@dataclass(frozen=True)
class EmpiricalDistributionTrajectory:
    action_shape: tuple[int, ...]
    horizons: np.ndarray
    vectors: np.ndarray

    @property
    def distributions(self) -> np.ndarray:
        return self.vectors.reshape((len(self.horizons), *self.action_shape), order="C")


def validate_action_shape(action_shape: Sequence[int]) -> tuple[int, ...]:
    try:
        shape = tuple(index(size) for size in action_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("action_shape must contain positive integers") from error
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("action_shape must contain one positive size per player")
    return shape


def default_checkpoints(horizon: int) -> np.ndarray:
    try:
        horizon = index(horizon)
    except TypeError as error:
        raise ValueError("horizon must be a positive integer") from error
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    checkpoints = [1]
    checkpoint = 100
    while checkpoint <= horizon:
        checkpoints.append(checkpoint)
        checkpoint *= 10
    checkpoints.append(horizon)
    return np.array(sorted(set(checkpoints)), dtype=int)


def final_logarithmic_interval_start(horizon: int) -> int:
    """Return the largest power of ten strictly smaller than ``horizon``."""
    try:
        horizon = index(horizon)
    except TypeError as error:
        raise ValueError("horizon must be a positive integer") from error
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    interval_start = 1
    while 10 * interval_start < horizon:
        interval_start *= 10
    return interval_start


def final_interval_checkpoints(
    horizon: int,
    final_interval_segments: int,
) -> np.ndarray:
    """Keep logarithmic history and subdivide only the final log interval."""
    try:
        horizon = index(horizon)
        final_interval_segments = index(final_interval_segments)
    except TypeError as error:
        raise ValueError(
            "horizon and final_interval_segments must be positive integers"
        ) from error
    if horizon <= 0 or final_interval_segments <= 0:
        raise ValueError(
            "horizon and final_interval_segments must be positive integers"
        )
    if horizon == 1:
        return np.array([1], dtype=int)

    interval_start = final_logarithmic_interval_start(horizon)
    checkpoints = []
    logarithmic = 1
    while logarithmic <= interval_start:
        checkpoints.append(logarithmic)
        logarithmic *= 10

    interval_width = horizon - interval_start
    for segment in range(1, final_interval_segments + 1):
        numerator = (
            interval_start * final_interval_segments
            + segment * interval_width
        )
        # Round to the nearest integer; exact halves round upward.
        endpoint = (
            numerator + final_interval_segments // 2
        ) // final_interval_segments
        if endpoint > checkpoints[-1]:
            checkpoints.append(endpoint)
    if checkpoints[-1] != horizon:
        checkpoints.append(horizon)
    return np.asarray(checkpoints, dtype=int)


def _validated_checkpoints(checkpoints: Iterable[int] | None, horizon: int) -> np.ndarray:
    if checkpoints is None:
        return default_checkpoints(horizon)
    try:
        values = [index(checkpoint) for checkpoint in checkpoints]
    except TypeError as error:
        raise ValueError("checkpoints must contain integers") from error
    if not values:
        raise ValueError("at least one checkpoint is required")
    if any(checkpoint <= 0 or checkpoint > horizon for checkpoint in values):
        raise ValueError(f"checkpoints must lie between 1 and {horizon}")
    return np.array(sorted(set(values)), dtype=int)


def empirical_distribution_trajectory(action_profiles: Iterable[Sequence[int]], action_shape: Sequence[int],
                                      checkpoints: Iterable[int] | None = None) -> EmpiricalDistributionTrajectory:
    shape = validate_action_shape(action_shape)
    profiles = list(action_profiles)
    if not profiles:
        raise ValueError("at least one action profile is required")
    horizons = _validated_checkpoints(checkpoints, len(profiles))
    checkpoint_indices = {int(horizon): position for position, horizon in enumerate(horizons)}
    counts = np.zeros(int(np.prod(shape)), dtype=np.int64)
    vectors = np.empty((len(horizons), counts.size), dtype=float)

    for horizon, raw_profile in enumerate(profiles, start=1):
        try:
            profile = tuple(index(action) for action in raw_profile)
        except TypeError as error:
            raise ValueError(f"round {horizon} must contain integer actions") from error
        if len(profile) != len(shape):
            raise ValueError(f"round {horizon} must contain one action per player")
        if any(action < 0 or action >= shape[player] for player, action in enumerate(profile)):
            raise ValueError(f"round {horizon} contains an out-of-range action")
        counts[np.ravel_multi_index(profile, shape, order="C")] += 1
        position = checkpoint_indices.get(horizon)
        if position is not None:
            vectors[position] = counts / horizon

    return EmpiricalDistributionTrajectory(shape, horizons, vectors)


def mean_empirical_distribution_trajectory(
    trajectories: Sequence[EmpiricalDistributionTrajectory],
) -> EmpiricalDistributionTrajectory:
    if not trajectories:
        raise ValueError("at least one empirical trajectory is required")
    first = trajectories[0]
    for trajectory in trajectories[1:]:
        if trajectory.action_shape != first.action_shape or not np.array_equal(trajectory.horizons, first.horizons):
            raise ValueError("empirical trajectories must have matching action shapes and horizons")
        if trajectory.vectors.shape != first.vectors.shape:
            raise ValueError("empirical trajectories must have matching vector shapes")
    vectors = np.mean([trajectory.vectors for trajectory in trajectories], axis=0)
    return EmpiricalDistributionTrajectory(first.action_shape, first.horizons.copy(), vectors)
