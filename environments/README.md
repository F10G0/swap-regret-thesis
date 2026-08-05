# Environments

Fixed repeated games and two isolated one-player stress-test environments.

Fixed-game payoffs use:

```text
payoff_tensor[player, action_player_0, ..., action_player_(n-1)]
```

There is one nonempty action dimension per player. Action counts may differ, and every payoff must be finite and in `[0, 1]`.

| Environment | Step | Feedback |
|---|---|---|
| `RepeatedGame` | `step(actions)` | Full deviation-payoff vector |
| `BanditRepeatedGame` | `step(actions)` | Selected payoff only |
| `HistoricalFrequencyAdversary` | `step((action,))` | Full payoff vector for evaluation |
| `LazyRandomWalkEnvironment` | `step()` | Precomputed action-independent reward vector |

Call `step()` before reading feedback. Bandit fixed-game runs use `deviation_payoffs()` only for offline regret evaluation; the learner never sees it.

The historical-frequency adversary constructs each payoff vector from earlier actions, using either full history or a positive memory window. Bandit mode passes only the sampled payoff to the learner.

The lazy random walk precomputes an independent integer-state walk for every action. Rewards lie on `0, 0.1, ..., 1`; initialization is centered at `0.5` or uniform on that grid. Its environment seed controls the sequence, which never depends on learner actions or the learner seed.
