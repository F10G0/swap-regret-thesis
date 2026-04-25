import numpy as np

from .ucb import UpperConfidenceBound


def make_ucb_standard(n_arms: int) -> UpperConfidenceBound:
    return UpperConfidenceBound(
        n_arms=n_arms,
        beta_fn=lambda t: 2.0 * np.log(max(t, 2)),
    )


def make_ucb_delta(n_arms: int, delta: float) -> UpperConfidenceBound:
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return UpperConfidenceBound(
        n_arms=n_arms,
        beta_fn=lambda _: 2.0 * np.log(1.0 / delta),
    )


def make_ucb_asymptotically_optimal(n_arms: int) -> UpperConfidenceBound:
    return UpperConfidenceBound(
        n_arms=n_arms,
        beta_fn=lambda t: 2.0
        * np.log(1.0 + max(t, 2) * (np.log(max(t, 2)) ** 2)),
    )
