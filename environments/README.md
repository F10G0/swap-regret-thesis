# Environments

Stateful fixed-payoff repeated-game environments. This package owns game state and feedback exposure, not learners or metrics.

## Payoff Tensor

```text
payoff_tensor[player, action_player_0, ..., action_player_(n-1)]
shape = (n_players, actions_player_0, ..., actions_player_(n-1))
```

Construction requires:

- at least one player and one action per player;
- one action dimension per player;
- finite payoffs in `[0, 1]`.

The tensor is copied. Heterogeneous action counts are available as `game.n_actions`.

## Round Interface

```python
game.step((action_player_0, ..., action_player_(n-1)))
feedback = game.feedback(player)
counterfactuals = game.deviation_payoffs(player)
```

`step()` validates and stores one joint action. `deviation_payoffs(player)` varies only that player's action and returns a copy of the unilateral-deviation payoff vector.

| Environment | `feedback(player)` |
|---|---|
| `RepeatedGame` | Complete unilateral-deviation payoff vector |
| `BanditRepeatedGame` | Realized scalar payoff |

In bandit experiments, `deviation_payoffs()` is used only by offline expected/realized regret evaluators and never reaches the learner.

The runner owns round sequencing; environments deliberately have no counter, learner state, duplicate-step guard, or payoff cache.

## Support and Ownership

The environment is generic over player count and heterogeneous actions. Custom 2–8 player games use the same classes as built-in games. N-player runs receive regret and equilibrium-convergence figures; matrix heatmaps remain a two-player presentation.

```text
environments/
├── base.py
└── repeated_game.py
```

Benchmark construction belongs to `experiments/`, learning to `algorithms/`, and analysis to `metrics/`.
