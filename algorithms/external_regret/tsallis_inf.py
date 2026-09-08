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

        # Scaled KKT equation: sum_i (a_i + z)^(-2) = 1,
        # a = eta * (L - min(L)). Its decreasing root lies in [1, sqrt(K)].
        a = self.learning_rate * (self.cumulative_loss - np.min(self.cumulative_loss))
        lower, upper = 1.0, np.sqrt(self.n_actions)
        z = (lower + upper) / 2.0
        inv = np.empty_like(a)
        probabilities = np.empty_like(a)

        # Newton normally converges in a handful of steps. The larger safety
        # ceiling also permits full-precision bisection near an endpoint root.
        tolerance = 1e-14
        for _ in range(64):
            np.add(a, z, out=inv)
            np.reciprocal(inv, out=inv)
            np.square(inv, out=probabilities)
            residual = np.sum(probabilities) - 1.0
            if abs(residual) <= tolerance or upper - lower <= tolerance * z:
                break
            if residual > 0.0:
                lower = z
            else:
                upper = z
            derivative = -2.0 * np.dot(probabilities, inv)
            newton = z - residual / derivative
            z = newton if np.isfinite(newton) and lower < newton < upper else (lower + upper) / 2.0
        else:
            raise FloatingPointError("Tsallis-INF normalizer did not converge")

        return probabilities / np.sum(probabilities)
