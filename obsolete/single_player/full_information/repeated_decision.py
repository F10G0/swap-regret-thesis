import numpy as np

from obsolete.single_player.base import FixedRewardEnvironment


class RepeatedDecision(FixedRewardEnvironment):
    """
    Full-information repeated decision problem.

    rewards[t, a] is the reward of action a at round t.

    At each round, the learner observes the entire reward vector.
    """

    def step(self, action: int) -> np.ndarray:
        self._validate_action(action)
        return self.next_rewards()

    def __repr__(self) -> str:
        return (
            f"RepeatedDecision("
            f"horizon={self.horizon}, "
            f"n_actions={self.n_actions})"
        )
