import numpy as np


RANDOM_WALK_GRID_MAX = 10
RANDOM_WALK_STEP = 1.0 / RANDOM_WALK_GRID_MAX
RANDOM_WALK_INITIALIZATIONS = ("centered", "uniform_grid")


class HistoricalFrequencyAdversary:
    """Punish the most frequent half of the actions over the full history."""

    def __init__(self, n_actions: int) -> None:
        self.n_players = 1
        self.n_actions = (n_actions,)
        self.action_counts = np.zeros(n_actions, dtype=int)
        self._tie_cursor = 0
        self.punished_actions = tuple(range((n_actions + 1) // 2))

    def step(self, actions: tuple[int, ...]) -> None:
        action = actions[0]
        n_actions = self.n_actions[0]
        tie_order = [(self._tie_cursor + offset) % n_actions for offset in range(n_actions)]
        punished_count = (n_actions + 1) // 2
        self.punished_actions = tuple(sorted(tie_order, key=self.action_counts.__getitem__, reverse=True)[:punished_count])
        self._tie_cursor = (self._tie_cursor + punished_count) % n_actions
        self.action_counts[action] += 1

    def feedback(self) -> np.ndarray:
        return self.deviation_payoffs()

    def deviation_payoffs(self) -> np.ndarray:
        payoffs = np.ones(self.n_actions[0])
        payoffs[list(self.punished_actions)] = 0.0
        return payoffs


class LazyRandomWalkEnvironment:
    """Action-independent fixed-grid reward random walks."""

    def __init__(self, n_actions: int, horizon: int, seed: int, initialization: str = "centered") -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if initialization not in RANDOM_WALK_INITIALIZATIONS:
            raise ValueError(f"unknown random-walk initialization: {initialization}")

        random = np.random.default_rng(seed)
        if initialization == "centered":
            states = np.full(n_actions, RANDOM_WALK_GRID_MAX // 2, dtype=int)
        else:
            states = random.integers(0, RANDOM_WALK_GRID_MAX + 1, size=n_actions)

        self.n_players = 1
        self.n_actions = (n_actions,)
        self.horizon = horizon
        self.reward_states = np.empty((horizon, n_actions), dtype=np.int8)
        self.reward_states[0] = states
        for time in range(1, horizon):
            for action in range(n_actions):
                states[action] = self._next_state(states[action], random.random())
            self.reward_states[time] = states
        self._round = -1

    @staticmethod
    def _next_state(state: int, draw: float) -> int:
        if state == 0:
            return 0 if draw < 0.5 else 1
        if state == RANDOM_WALK_GRID_MAX:
            return state if draw < 0.5 else state - 1
        if draw < 1.0 / 3.0:
            return state - 1
        if draw < 2.0 / 3.0:
            return state
        return state + 1

    def step(self) -> None:
        if self._round + 1 >= self.horizon:
            raise RuntimeError("random-walk horizon is exhausted")
        self._round += 1

    def feedback(self) -> np.ndarray:
        if self._round < 0:
            raise RuntimeError("call step before requesting feedback")
        return self.reward_states[self._round] * RANDOM_WALK_STEP
