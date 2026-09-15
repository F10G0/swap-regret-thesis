from collections.abc import Sequence
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
