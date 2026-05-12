import math

from .elimination import EliminationAlgorithm


def make_elimination(n_arms: int, phase_length_fn) -> EliminationAlgorithm:
    return EliminationAlgorithm(
        n_arms=n_arms,
        phase_length_fn=phase_length_fn,
    )


def make_elimination_standard(n_arms: int, delta: float) -> EliminationAlgorithm:
    """
    Standard phased elimination algorithm.

    Uses phase length:
        m_l = ceil(2^(2l) * log(1 / delta))
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")

    def phase_length_fn(phase: int) -> int:
        return math.ceil((2.0 ** (2 * phase)) * math.log(1.0 / delta))

    return make_elimination(
        n_arms=n_arms,
        phase_length_fn=phase_length_fn,
    )
