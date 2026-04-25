import math

from .etc import ExploreThenCommit
from .doubling_trick import DoublingTrickWrapper


def optimal_etc_m(n_arms: int, horizon: int) -> int:
    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    return math.ceil((horizon / n_arms) ** (2.0 / 3.0))


def make_etc(n_arms: int, m: int) -> ExploreThenCommit:
    return ExploreThenCommit(
        n_arms=n_arms,
        m=m,
    )


def make_etc_doubling(n_arms: int, m_fn=None, initial_epoch_length: int = 1) -> DoublingTrickWrapper:
    """
    Doubling-trick wrapper for ETC.

    By default, uses the theoretical choice:
        m(T) = ceil((T / K)^(2/3))
    where T is the epoch length.
    """
    if m_fn is None:
        m_fn = lambda T: optimal_etc_m(n_arms, T)

    def algorithm_factory(n_arms: int, epoch: int, epoch_length: int):
        return make_etc(
            n_arms=n_arms,
            m=m_fn(epoch_length),
        )

    return DoublingTrickWrapper(
        n_arms=n_arms,
        algorithm_factory=algorithm_factory,
        initial_epoch_length=initial_epoch_length,
    )
