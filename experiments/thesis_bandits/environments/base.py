from abc import ABC, abstractmethod


class BanditEnvironment(ABC):
    """
    Abstract base class for bandit environments.

    Defines the interface that all environments must implement.
    """

    @property
    @abstractmethod
    def n_arms(self) -> int:
        """
        Return the number of available actions (arms).
        """
        pass

    @abstractmethod
    def pull(self, action: int) -> float:
        """
        Execute the given action and return a reward.

        Args:
            action (int): index of the chosen arm

        Returns:
            float: sampled reward
        """
        pass

    @abstractmethod
    def arm_mean(self, action: int) -> float:
        """
        Return the true expected reward of the given arm.

        NOTE:
        This should NOT be used by algorithms, only for evaluation.

        Args:
            action (int)

        Returns:
            float
        """
        pass

    @abstractmethod
    def optimal_mean(self) -> float:
        """
        Return the expected reward of the optimal arm.
        """
        pass

    @abstractmethod
    def optimal_arm(self) -> int:
        """
        Return the index of the optimal arm.
        """
        pass

    def reset(self) -> None:
        """
        Reset environment state (optional).
        Default: stateless, so do nothing.
        """
        pass
