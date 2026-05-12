import numpy as np

from .base import StochasticBanditEnvironment


class BernoulliBandit(StochasticBanditEnvironment):
    """
    Stochastic multi-armed bandit with Bernoulli rewards.

    Each arm a has reward distribution:
        X ~ Bernoulli(p_a)
    """

    def __init__(self, probs: list[float]):
        if len(probs) == 0:
            raise ValueError("probs must not be empty")

        for p in probs:
            if p < 0.0 or p > 1.0:
                raise ValueError("probabilities must be in [0, 1]")

        self._probs = np.array(probs, dtype=float)

    @property
    def n_arms(self) -> int:
        return len(self._probs)

    def pull(self, action: int) -> float:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")

        return float(np.random.binomial(1, self._probs[action]))

    def arm_mean(self, action: int) -> float:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")

        return float(self._probs[action])

    def optimal_mean(self) -> float:
        return float(np.max(self._probs))

    def optimal_arm(self) -> int:
        return int(np.argmax(self._probs))

    def __repr__(self) -> str:
        return f"BernoulliBandit(probs={self._probs.tolist()})"
