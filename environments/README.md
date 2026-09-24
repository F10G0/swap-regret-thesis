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

Runners supply valid actions/player ids and stop at the configured horizon; the per-round environment methods trust that call sequence. Fixed-game constructors still validate and copy external payoff tensors. Deviation-payoff vectors are views of that owned tensor and must be consumed read-only.

The historical-frequency adversary uses cumulative action counts from earlier rounds, breaking ties with a rotating order. It assigns payoff 0 to the most frequent `ceil(K/2)` actions and payoff 1 to the others. Bandit mode passes only the sampled payoff to the learner.

The environment precomputes an independent lazy random walk for every action. Rewards lie on `0, 0.1, ..., 1`, and every walk is initialized at `0.5`. Its random stream is derived from the experiment's base seed in a separate domain from the learner and never depends on learner actions.
