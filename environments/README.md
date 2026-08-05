# Environments

The package contains fixed-payoff repeated games and one isolated adaptive stress-test environment.

## Payoff Tensor

```text
payoff_tensor[player, action_player_0, ..., action_player_(n-1)]
```

The tensor must contain one action dimension per player, at least one action per player, and finite payoffs in `[0, 1]`. Different players may have different action counts.

## Round Interfaces

Fixed games use:

```python
game.step(actions)
feedback = game.feedback(player)
deviation_payoffs = game.deviation_payoffs(player)
```

The one-player adversary uses:

```python
environment.step((action,))
feedback = environment.feedback()
```

Call `step()` before requesting feedback in either case.

| Environment | Learner feedback |
|---|---|
| `RepeatedGame` | Full unilateral-deviation payoff vector |
| `BanditRepeatedGame` | Realized scalar payoff |
| `HistoricalFrequencyAdversary` | Full payoff vector; the most frequent action in its memory receives 0 and all others receive 1 |

In bandit runs, `deviation_payoffs()` is used only by the offline regret evaluator and is never passed to the learner.

`HistoricalFrequencyAdversary` has one learner. Its memory may cover any positive number of previous rounds or the full history. It constructs the payoff vector before adding the current action and rotates ties deterministically.

The experiment runner owns round counting and learner updates. Game construction belongs to `experiments/`, learning to `algorithms/`, and analysis to `metrics/`.
