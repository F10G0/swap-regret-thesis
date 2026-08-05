# Algorithms

Regret-minimizing learners used by the experiment runner.

## Available Learners

| Name | Feedback | Purpose |
|---|---|---|
| `hedge` | Reward vector | External regret |
| `exp3` | Realized reward | Bandit external regret |
| `exp3_ix` | Realized reward | External regret with implicit exploration |
| `regret_matching` | Reward vector | Internal regret |
| `stationary_regret_matching` | Reward vector | Internal regret using a stationary distribution |
| `bm` | Mode-dependent | Swap-regret reduction |
| `ito` | Mode-dependent | Swap-regret reduction |
| `lce_ix` | Realized reward | Bandit swap-regret reduction |

## Interface

Every learner follows the same lifecycle:

```python
strategy = learner.strategy()
action = learner.sample_action()
learner.update(feedback)
```

Strategies are finite probability vectors. Full-information learners receive a reward vector; bandit learners receive one realized reward. `reset()` restores the initial uniform strategy.

The experiment layer creates one learner per player and gives each learner a deterministic seed.

## Learning Rates

Hedge, Exp3, and Exp3-IX use the configured horizon in their learning-rate schedule. Direct construction with `horizon=0` gives an anytime schedule; experiment runs require a positive horizon.

The shared exponential-weights implementation uses a stable softmax and floors probabilities at `NUMERICAL_TOLERANCE`.

## Regret Reductions

- Regret Matching uses positive action-replacement regret.
- SRM uses a stationary distribution of the regret transition matrix.
- Blum–Mansour updates one external-regret learner per outer action.
- Ito updates one sampled inner learner per round.
- LCE-IX uses Exp3-IX-based bandit updates.

`stationary_distribution(...)` supports `solve`, `pinv`, and `iteration`; the project default is `solve`.

```text
algorithms/
├── base.py
├── stationary.py
├── external_regret/
├── internal_regret/
└── swap_regret/
```

Games and feedback belong to `environments/`; experiment construction to `experiments/`; regret measurement to `metrics/`.
