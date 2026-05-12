import numpy as np

from .phased_ucb import PhasedUCB


def make_phased_ucb_exponential(n_arms: int, delta: float) -> PhasedUCB:
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return PhasedUCB(
        n_arms=n_arms,
        beta_fn=lambda _: 2.0 * np.log(1.0 / delta),
        phase_length_fn=lambda phase, arm, counts, t: 2 ** phase,
    )


def make_phased_ucb_count_doubling(n_arms: int, delta: float, alpha: float = 2.0) -> PhasedUCB:
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1")

    def phase_length_fn(phase, arm, counts, t):
        current_count = counts[arm]
        return np.ceil((alpha - 1.0) * current_count)

    return PhasedUCB(
        n_arms=n_arms,
        beta_fn=lambda _: 2.0 * np.log(1.0 / delta),
        phase_length_fn=phase_length_fn,
    )
