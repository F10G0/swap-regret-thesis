import numpy as np

from algorithms.base import Algorithm


class ExponentialWeightsAlgorithm(Algorithm):
    """
    Base class for exponential-weights algorithms.
    """

    def __init__(self, n_actions: int, learning_rate: float, seed: int | None = None) -> None:
        super().__init__(n_actions, seed)

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.learning_rate = learning_rate
        self.reset()

    def _exponential_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute the exponential-weights distribution induced by scores.
        """
        logits = self.learning_rate * scores

        # Numerical stability.
        logits -= np.max(logits)
        weights = np.exp(logits)
        return weights / np.sum(weights)
