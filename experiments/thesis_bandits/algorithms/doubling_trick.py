from .base import BanditAlgorithm


class DoublingTrickWrapper(BanditAlgorithm):
    """
    Generic doubling-trick wrapper for bandit algorithms.

    The wrapper restarts the base algorithm in epochs whose lengths grow
    exponentially. This can turn horizon-dependent algorithms into anytime
    algorithms.
    """

    def __init__(self, n_arms: int, algorithm_factory, initial_epoch_length: int = 1):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if algorithm_factory is None:
            raise ValueError("algorithm_factory must not be None")
        if initial_epoch_length <= 0:
            raise ValueError("initial_epoch_length must be positive")
        self.n_arms = n_arms
        self.algorithm_factory = algorithm_factory
        self.initial_epoch_length = initial_epoch_length
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.epoch = 0
        self.epoch_t = 0
        self.epoch_length = self.initial_epoch_length
        self.algorithm = self.algorithm_factory(
            n_arms=self.n_arms,
            epoch=self.epoch,
            epoch_length=self.epoch_length,
        )

    def select_action(self) -> int:
        return self.algorithm.select_action()

    def update(self, action: int, reward: float) -> None:
        self.algorithm.update(action, reward)
        self.t += 1
        self.epoch_t += 1
        if self.epoch_t >= self.epoch_length:
            self._advance_epoch()

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.epoch_t = 0
        self.epoch_length *= 2
        self.algorithm = self.algorithm_factory(
            n_arms=self.n_arms,
            epoch=self.epoch,
            epoch_length=self.epoch_length,
        )

    def __repr__(self) -> str:
        return (
            f"DoublingTrickWrapper("
            f"n_arms={self.n_arms}, "
            f"t={self.t}, "
            f"epoch={self.epoch}, "
            f"epoch_t={self.epoch_t}, "
            f"epoch_length={self.epoch_length}, "
            f"algorithm={self.algorithm}"
            f")"
        )
    