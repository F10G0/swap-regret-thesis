# Algorithms

Regret-minimizing learners used by the repeated-game runner.

## Available Learners

| Registry name | Feedback | Objective |
|---|---|---|
| `hedge` | Reward vector | External regret |
| `exp3` | Realized reward | External regret |
| `exp3_ix` | Realized reward | External regret with implicit exploration |
| `regret_matching` | Reward vector | Internal regret |
| `stationary_regret_matching` | Reward vector | Internal regret |
| `bm` | Mode-dependent | Swap-regret reduction |
| `ito` | Mode-dependent | Swap-regret reduction |
| `lce_ix` | Realized reward | Bandit swap-regret reduction |

The full-information and bandit registries map `bm` and `ito` to their corresponding classes. The UI and figures abbreviate Regret Matching as RM and Stationary Regret Matching as SRM.

Regret evaluation is not a learner input. Expected, realized, or both changes only the offline metrics recorded by the runner.

## Lifecycle

Every learner derives from `Algorithm`:

```python
action = learner.sample_action()
learner.update(feedback)
strategy = learner.strategy()
```

- `n_actions` must be positive and `horizon` non-negative.
- `reset()` restores uniform strategy and clears state.
- `sample_action()` stores the sampled action.
- `update()` validates feedback, updates state, advances local time, and validates the new strategy.
- `strategy()` returns a copy.

Strategies must be finite, non-negative, correctly shaped, and normalized. Full-information feedback is a reward vector in `[0, 1]`; bandit feedback is one realized reward.

The experiment layer creates one learner per player and derives distinct deterministic seeds.

## Learning Rates

Exponential-weights learners use:

```text
rate_horizon = max(configured_horizon, t + 1)
```

Thus a positive horizon gives a known-horizon rate and `horizon=0` gives an anytime rate.

| Learner | Update |
|---|---|
| Hedge | `eta = sqrt(8 log(K) / rate_horizon)` |
| Exp3 | `eta = sqrt(log(K) / (K * rate_horizon))` and importance-weighted reward |
| Exp3-IX | `eta = sqrt(log(K) / rate_horizon)`, `gamma = eta / 2`, implicit-exploration loss |

The shared stable softmax floors probabilities at `NUMERICAL_TOLERANCE` before renormalizing.

## Internal Regret

Regret Matching maintains:

```text
R[a_t, :] += reward_vector - reward_vector[a_t]
```

Positive regret forms a row-stochastic transition matrix.

- RM uses the transition row of the latest sampled action.
- SRM uses a stationary distribution of the complete transition matrix.

## Swap-Regret Reductions

Each reduction owns one external-regret learner per outer action. Their strategies form a transition matrix whose stationary distribution is the outer strategy.

- **BM:** updates every inner learner; Hedge under full information and Exp3 under bandit feedback.
- **Ito:** samples and updates one anytime inner learner per round.
- **LCE-IX:** updates anytime Exp3-IX learners with transformed bandit losses.

## Stationary Distributions

`stationary_distribution(transition_matrix, method=...)` supports:

- `solve`: direct solve with pseudoinverse fallback;
- `pinv`: pseudoinverse;
- `iteration`: fixed-point iteration from uniform.

Inputs must be finite row-stochastic square matrices. Outputs use `NUMERICAL_TOLERANCE = 1e-10` for bounds, normalization, and `||pQ - p||_1`.

## Structure and Ownership

```text
algorithms/
├── base.py
├── stationary.py
├── external_regret/
├── internal_regret/
└── swap_regret/
```

This package owns learner state and stationary computation. Games and feedback belong to `environments/`; construction and seeding to `experiments/`; regret measurement to `metrics/regret.py`.
