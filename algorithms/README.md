# Algorithms

Regret-minimizing learners used by the experiment runners.

| Name | Feedback | Objective |
|---|---|---|
| `hedge` | Reward vector | External regret |
| `exp3`, `exp3_ix` | Realized reward | Bandit external regret |
| `regret_matching`, `stationary_regret_matching` | Reward vector | Internal regret |
| `bm`, `ito` | Mode-dependent | Swap regret |
| `lce_ix` | Realized reward | Bandit swap regret |

Every learner follows the same lifecycle:

```python
strategy = learner.strategy()
action = learner.sample_action()
learner.update(feedback)
```

Strategies are probability vectors. Full-information learners receive a reward vector; bandit learners receive only the selected reward. `reset()` restores the initial uniform strategy. Experiment runners create one deterministically seeded learner per player.

Hedge, Exp3, and Exp3-IX use the configured horizon for their learning rate. Direct construction with `horizon=0` uses an anytime schedule, while experiments require a positive horizon. Exponential weights use a stable softmax and the probability floor from `config.py`.

The reductions are:

- Regret Matching: positive action-replacement regret.
- SRM: a stationary distribution of the regret transition matrix.
- Blum–Mansour: one external learner per outer action.
- Ito: one sampled inner learner per round.
- LCE-IX: Exp3-IX-based bandit updates.

The SRM solver supports `solve`, `pinv`, and `iteration`; `solve` is the default. Games and feedback live in `environments/`, orchestration in `experiments/`, and evaluation in `metrics/`.
