from abc import ABC, abstractmethod


class BanditEnvironment(ABC):
    @property
    @abstractmethod
    def n_arms(self) -> int:
        pass

    @abstractmethod
    def pull(self, action: int) -> float:
        pass

    def reset(self) -> None:
        pass


class StochasticBanditEnvironment(BanditEnvironment):
    @abstractmethod
    def arm_mean(self, action: int) -> float:
        pass

    @abstractmethod
    def optimal_mean(self) -> float:
        pass

    @abstractmethod
    def optimal_arm(self) -> int:
        pass


class AdversarialBanditEnvironment(BanditEnvironment):
    @abstractmethod
    def best_fixed_arm_reward(self) -> float:
        """
        Total reward of the best fixed arm in hindsight.
        Used for external regret evaluation.
        """
        pass
    