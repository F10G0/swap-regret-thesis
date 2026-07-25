# Algorithms

Regret-minimization algorithms used by the repeated-game experiments.

## Structure

```text
algorithms/
├── __init__.py
├── base.py
├── stationary.py
├── external_regret/
│   ├── __init__.py
│   ├── base.py
│   ├── exp3.py
│   └── hedge.py
├── internal_regret/
│   ├── __init__.py
│   ├── base.py
│   └── regret_matching.py
└── swap_regret/
    ├── __init__.py
    ├── base.py
    ├── blum_mansour.py
    ├── ito.py
    └── lce_ix.py
```

## Common Lifecycle

`Algorithm` owns the action count, optional horizon, random generator, local update count, current action, and current strategy.

```python
action = learner.sample_action()
learner.update(feedback)
```

- `reset()` restores algorithm-specific state, sets `t = 0`, clears `current_action`, and installs the uniform strategy.
- `sample_action()` samples from `current_strategy` and normally stores the result in `current_action`.
- `update(feedback)` validates the reward feedback, updates algorithm-specific state, increments `t`, computes the next strategy, and validates the resulting distribution.
- `strategy()` returns a copy of the current action distribution.

The experiment runner guarantees that an ordinary learner is sampled before it is updated. Bandit BM and LCE-IX explicitly assign the outer sampled action to their inner learners before forwarding transformed feedback. Ito instead samples through one selected inner learner, whose own `sample_action()` stores the final action.

## Horizon-Dependent Rates

Learning rates are internal algorithm properties; experiments never pass a learning rate.

Exponential-weights learners use

```text
rate_horizon = max(configured_horizon, t + 1).
```

Consequently:

- `horizon > 0` uses the theoretically motivated known-horizon rate through the configured horizon and continues safely with the elapsed-round horizon if execution runs longer;
- `horizon == 0` gives an anytime schedule based entirely on the learner's local update count.

The local count matters for reductions such as Ito and LCE-IX, where an inner learner's effective horizon is not known in advance.

## External Regret

### `external_regret/base.py`

`ExponentialWeightsAlgorithm` provides shared cumulative-score state and a numerically stable softmax computation. The resulting probabilities are floored at `NUMERICAL_TOLERANCE` and renormalized to prevent floating-point underflow from producing zero importance-weight denominators. Subclasses define their learning-rate property and feedback update.

### Hedge

`Hedge` receives a full reward vector and updates cumulative rewards:

```text
eta = sqrt(8 log(K) / rate_horizon).
```

### Exp3

`Exp3` receives one realized reward and applies the standard importance-weighted reward estimate to the sampled action:

```text
eta = sqrt(log(K) / (K * rate_horizon)).
```

### Exp3-IX

`Exp3IX` also receives one realized reward, converts it to a loss internally, and applies implicit exploration:

```text
eta   = sqrt(log(K) / rate_horizon)
gamma = eta / 2
```

Its cumulative score is the negative estimated cumulative loss. Exp3 and Exp3-IX share the exponential-weights strategy computation but keep separate update rules.

## Internal Regret

### `internal_regret/base.py`

`RegretMatchingBase` maintains the cumulative pairwise-regret matrix. On each round it updates only the row corresponding to the action sampled in that round:

```text
R[a_t, :] += reward_vector - reward_vector[a_t].
```

Its regret-induced transition matrix scales the positive cumulative regrets by `K * t` and assigns each remaining row probability to the diagonal.

### Regret Matching

`RegretMatching` is the inertia-based Hart–Mas-Colell procedure from equation (2.2). Its next strategy is the transition row corresponding to the most recently sampled action.

### Stationary Regret Matching

`StationaryRegretMatching` is the invariant-distribution procedure from equation (3.1). It computes a stationary distribution of the complete regret-induced transition matrix.

The two procedures share the same realized pairwise-regret update and differ only in strategy computation.

## Swap Regret

### `swap_regret/base.py`

`StationaryReduction` manages one external-regret learner per action. Their current strategies form a row-stochastic transition matrix, and the outer strategy is one stationary distribution of that matrix. Inner learners receive reproducible seeds drawn from the outer learner's random generator.

### Blum–Mansour

- `FullBM` uses known-horizon Hedge learners and updates every inner learner with its weighted full-information reward vector.
- `BanditBM` uses known-horizon Exp3 learners and updates every inner learner with its importance-weighted observed reward.

Because every inner learner is updated each round, the experiment horizon is also the local inner horizon.

### Ito

- `FullIto` uses anytime Hedge learners.
- `BanditIto` uses anytime Exp3 learners.

Ito first samples an inner learner from the outer stationary distribution and then samples the final action from that learner. Only the selected learner is updated, so its local horizon is unknown and it is constructed with `horizon=0`.

### LCE-IX

`LCEIX` is the bandit-feedback learning-for-correlated-equilibrium reduction with implicit exploration. It uses one anytime Exp3-IX learner per action, transforms the realized outer reward into each learner's weighted observed loss, and updates every inner learner. The inner schedules use their local counts:

```text
eta_t   = sqrt(log(K) / (t + 1))
gamma_t = eta_t / 2
```

Here the displayed `t + 1` is the rate horizon used while processing the next update.

## Stationary Distributions

`stationary.py` provides three selectable methods:

- `solve`: direct linear solve, with automatic pseudoinverse fallback when the system is singular or numerically invalid;
- `pinv`: pseudoinverse solve;
- `iteration`: fixed-point iteration.

The default is configured by `STATIONARY_METHOD` in `config.py`. Every result is checked for shape, finiteness, probability bounds, normalization, and the stationarity residual `||pQ - p||_1`.
