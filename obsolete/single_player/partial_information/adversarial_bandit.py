import numpy as np

from obsolete.single_player.base import FixedRewardEnvironment


class AdversarialBandit(FixedRewardEnvironment):
    """
    Adversarial bandit problem.

    rewards[t, a] is the reward of action a at round t.

    At each round, the learner observes only the reward
    of the selected action.
    """

    def step(self, action: int) -> float:
        self._validate_action(action)
        return float(self.next_rewards()[action])

    def pull(self, arm: int) -> float:
        """
        Alias for step() using standard bandit terminology.
        """
        return self.step(arm)

    def __repr__(self) -> str:
        return (
            f"AdversarialBandit("
            f"horizon={self.horizon}, "
            f"n_actions={self.n_actions})"
        )
