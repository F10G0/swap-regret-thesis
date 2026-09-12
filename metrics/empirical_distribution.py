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
    """Accumulate project-generated or loader-validated joint-action profiles."""
    shape = validate_action_shape(action_shape)
    profiles = list(action_profiles)
    if not profiles:
        raise ValueError("at least one action profile is required")
    horizons = _validated_checkpoints(checkpoints, len(profiles))
    checkpoint_indices = {int(horizon): position for position, horizon in enumerate(horizons)}
    counts = np.zeros(int(np.prod(shape)), dtype=np.int64)
    vectors = np.empty((len(horizons), counts.size), dtype=float)

    for horizon, profile in enumerate(profiles, start=1):
        counts[np.ravel_multi_index(profile, shape, order="C")] += 1
        position = checkpoint_indices.get(horizon)
        if position is not None:
            vectors[position] = counts / horizon

    return EmpiricalDistributionTrajectory(shape, horizons, vectors)
