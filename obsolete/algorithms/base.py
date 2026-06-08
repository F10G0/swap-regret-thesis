from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """
    Abstract base class for bandit algorithms.

    All algorithms should implement a common interface so that they can be
    used interchangeably in experiments and future integrations.
    """

    @abstractmethod
    def select_action(self) -> int:
        """
        Select an action (arm) for the current round.

        Returns:
            int: index of the chosen arm
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the internal state after observing the reward.

        Args:
            action (int): chosen arm
            reward (float): observed reward
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the algorithm.
        """
        pass
