from collections import deque

import numpy as np


class HistoricalFrequencyAdversary:
    """One-player environment that punishes the most-played past action."""

    def __init__(self, n_actions: int, memory_window: int | None = None) -> None:
        self.n_players = 1
        self.n_actions = (n_actions,)
        self.action_counts = np.zeros(n_actions, dtype=int)
        self._history = None if memory_window is None else deque(maxlen=memory_window)
        self._counts = self.action_counts if self._history is None else np.zeros(n_actions, dtype=int)
        self._tie_cursor = 0
        self.punished_action = 0

    def step(self, actions: tuple[int, ...]) -> None:
        action = actions[0]
        n_actions = self.n_actions[0]
        maximum_count = self._counts.max()
        punished_action = self._tie_cursor
        while self._counts[punished_action] != maximum_count:
            punished_action = (punished_action + 1) % n_actions
        self.punished_action = punished_action
        self._tie_cursor = (punished_action + 1) % n_actions

        self.action_counts[action] += 1
        if self._history is not None:
            if len(self._history) == self._history.maxlen:
                self._counts[self._history[0]] -= 1
            self._history.append(action)
            self._counts[action] += 1

    def feedback(self) -> np.ndarray:
        return self.deviation_payoffs()

    def deviation_payoffs(self) -> np.ndarray:
        payoffs = np.ones(self.n_actions[0])
        payoffs[self.punished_action] = 0.0
        return payoffs
