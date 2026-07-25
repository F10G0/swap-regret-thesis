# Environments

Stateful fixed-payoff repeated-game environments.

## Structure

```text
environments/
├── __init__.py
├── base.py
└── repeated_game.py
```

The environment layer is independent of `algorithms`; it represents the game and exposes feedback without owning player learners.

## Payoff Representation

Each environment receives one NumPy payoff tensor:

```text
payoff_tensor[player, action_0, ..., action_n]
```

Its shape is `(n_players, action_1, ..., action_n)`. The tuple of action counts is derived from the joint-action dimensions, and the number of players is derived from the leading dimension.

Construction validates the structural invariants needed by the framework:

- the leading player count matches the number of action dimensions;
- every player has at least one action;
- every payoff is finite and lies in `[0, 1]`.

The tensor is copied on construction so later changes to the caller's array cannot alter the game.

## Round Interface

```python
game.step(actions)
feedback = game.feedback(player)
```

`step(actions)` validates and stores one complete joint action. The runner guarantees exactly one call per round before any feedback or evaluation query. The environment intentionally does not add round counters, duplicate-step checks, or cached derived state.

`deviation_payoffs(player)` returns a copy of the payoff vector obtained by varying only the requested player's action while holding the other current actions fixed.

## Feedback Models

### `RepeatedGame`

Full-information feedback returns the requested player's complete unilateral-deviation payoff vector. The realized payoff is the component corresponding to that player's selected action.

### `BanditRepeatedGame`

Bandit feedback performs one scalar tensor lookup and returns only the requested player's realized payoff. It does not compute or store a counterfactual payoff vector.

The experiment runner may call `deviation_payoffs(player)` separately to evaluate realized regret. This is evaluation-only information and is never passed to a bandit learner.

## Computation and State

The shared state consists only of:

- the validated payoff tensor;
- the current joint action after `step()`.

Realized payoffs and deviation vectors are queried separately for each player. Nothing is precomputed or cached, keeping the two feedback models simple and ensuring bandit learners observe only bandit feedback.
