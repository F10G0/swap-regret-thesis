import numpy as np

from algorithms.base import Algorithm


class TsallisINF(Algorithm):
    """Anytime 1/2-Tsallis-INF with importance-weighted bandit losses."""

    @property
    def learning_rate(self) -> float:
        """Return eta_t = 1 / sqrt(t) for the next local round."""
        return 1.0 / np.sqrt(self.t + 1.0)

    def _reset_state(self) -> None:
        self.cumulative_loss = np.zeros(self.n_actions, dtype=float)

    def _update_state(self, reward: float) -> None:
        loss = 1.0 - reward
        probability = self.current_strategy[self.current_action]
        self.cumulative_loss[self.current_action] += loss / probability

    def _compute_strategy(self) -> np.ndarray:
        if self.n_actions == 1:
            return np.ones(1, dtype=float)

        # The KKT conditions for the 1/2-Tsallis regularizer give
        #   w_i = 1 / (eta^2 (L_i + lambda)^2).
        # Shift cumulative losses for translation invariance, then find the
        # unique positive normalizer that makes the probabilities sum to one.
        shifted_loss = self.cumulative_loss - np.min(self.cumulative_loss)
        eta = self.learning_rate
        lower = 0.0
        upper = np.sqrt(self.n_actions) / eta

        for _ in range(100):
            normalizer = (lower + upper) / 2.0
            probabilities = 1.0 / np.square(eta * (shifted_loss + normalizer))
            if np.sum(probabilities) > 1.0:
                lower = normalizer
            else:
                upper = normalizer

        probabilities = 1.0 / np.square(eta * (shifted_loss + upper))
        return probabilities / np.sum(probabilities)
