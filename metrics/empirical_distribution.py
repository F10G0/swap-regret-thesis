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


def uniform_checkpoints(horizon: int, count: int) -> np.ndarray:
    try:
        horizon = index(horizon)
        count = index(count)
    except TypeError as error:
        raise ValueError("horizon and count must be positive integers") from error
    if horizon <= 0 or count <= 0:
        raise ValueError("horizon and count must be positive integers")
    count = min(horizon, count)
    return np.rint(np.linspace(1, horizon, count)).astype(int)


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
