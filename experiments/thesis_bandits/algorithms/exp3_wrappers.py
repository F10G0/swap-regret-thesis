import math

from .exp3 import Exp3
from .exp3_adaptive import Exp3AdaptiveLR
from .doubling_trick import DoublingTrickWrapper


def optimal_exp3_learning_rate(n_arms: int, horizon: int) -> float:
    return math.sqrt(2 * math.log(n_arms) / (horizon * n_arms))


def optimal_adaptive_exp3_learning_rate(n_arms: int, t: int) -> float:
    return math.sqrt(math.log(n_arms) / (t * n_arms))


def make_exp3(n_arms: int, learning_rate: float, implicit_exploration: float = 0.0) -> Exp3:
    """
    Fully manual Exp3 / Exp3-IX.

    implicit_exploration = 0.0 -> Exp3
    implicit_exploration > 0.0 -> Exp3-IX
    """

    return Exp3(
        n_arms=n_arms,
        learning_rate=learning_rate,
        implicit_exploration=implicit_exploration,
    )


def _exp3_factory(ix_ratio: float):
    """
    Create a factory for DoublingTrickWrapper.

    ix_ratio:
        0.0 -> Exp3
        0.5 -> Exp3-IX with implicit_exploration = learning_rate / 2
    """

    def factory(n_arms: int, epoch: int, epoch_length: int):
        learning_rate = optimal_exp3_learning_rate(n_arms, epoch_length)
        implicit_exploration = ix_ratio * learning_rate
        return Exp3(
            n_arms=n_arms,
            learning_rate=learning_rate,
            implicit_exploration=implicit_exploration,
        )

    return factory


def make_exp3_doubling(n_arms: int) -> DoublingTrickWrapper:
    return DoublingTrickWrapper(
        n_arms=n_arms,
        algorithm_factory=_exp3_factory(ix_ratio=0.0),
    )


def make_exp3_ix_doubling(n_arms: int) -> DoublingTrickWrapper:
    return DoublingTrickWrapper(
        n_arms=n_arms,
        algorithm_factory=_exp3_factory(ix_ratio=0.5),
    )


def make_exp3_adaptive(n_arms: int) -> Exp3AdaptiveLR:
    return Exp3AdaptiveLR(
        n_arms=n_arms,
        learning_rate_fn=optimal_adaptive_exp3_learning_rate,
        ix_ratio=0.0,
    )


def make_exp3_ix_adaptive(n_arms: int) -> Exp3AdaptiveLR:
    return Exp3AdaptiveLR(
        n_arms=n_arms,
        learning_rate_fn=optimal_adaptive_exp3_learning_rate,
        ix_ratio=0.5,
    )
