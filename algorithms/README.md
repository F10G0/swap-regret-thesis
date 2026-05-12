# Algorithms Module

This module contains the implementations of online learning and bandit algorithms used throughout the thesis project.

The framework is designed around:
- unified algorithm interfaces,
- modular implementations,
- reusable shared components,
- extensibility toward stronger regret notions.

The current focus is on no-external-regret algorithms in stochastic and adversarial bandit settings, which will later serve as building blocks for reduction-based no-swap-regret algorithms.

---

# Design Philosophy

The framework separates:
- core algorithm logic,
- shared reusable components,
- public wrappers,
- experiment-facing interfaces.

This separation is intended to:
- simplify experimentation,
- reduce duplicated logic,
- improve extensibility,
- make future regret reductions easier to implement.

In particular, many no-swap-regret algorithms are constructed by wrapping multiple no-external-regret learners.
Therefore, reusable external learners are treated as first-class components in the framework design.

---

# Unified Interface

All algorithms follow a unified interface:

```python
select_action()
update(action, reward)
reset()
```

This abstraction allows algorithms to be:
- exchanged easily in experiments,
- benchmarked under identical environments,
- wrapped by higher-level reductions.

The interface is intentionally minimal to support both:
- stochastic bandit algorithms,
- adversarial learning algorithms.

---

# Shared Components

## empirical_mean_base.py

Several stochastic algorithms maintain:
- pull counts,
- reward sums,
- empirical means.

These shared statistics are abstracted into:

```text
EmpiricalMeanBanditAlgorithm
```

to avoid duplicated logic across:
- ETC,
- UCB,
- phased UCB,
- elimination-based methods.

---

## doubling_trick.py

Some algorithms require horizon-dependent parameters.

The framework therefore provides a generic doubling-trick wrapper that:
- restarts algorithms periodically,
- increases epoch lengths exponentially,
- converts horizon-dependent algorithms into anytime algorithms.

The wrapper is implemented independently from concrete algorithms in order to maximize reuse.

---

# Implemented Algorithms

## Explore-Then-Commit (ETC)

A simple stochastic bandit baseline:
1. explore each arm uniformly,
2. estimate empirical means,
3. commit permanently to the empirically best arm.

The implementation also supports:
- doubling-trick wrappers,
- configurable exploration lengths.

---

## Upper Confidence Bound (UCB)

Optimism-based stochastic bandit algorithms.

The framework currently includes:
- standard UCB,
- delta-based variants,
- asymptotically optimal variants,
- phased UCB.

The implementations emphasize:
- modular confidence schedules,
- reusable exploration logic,
- configurable phase structures.

---

## Elimination-Based Algorithms

Phase-based stochastic algorithms that:
- explore active arms,
- eliminate statistically suboptimal arms,
- eventually commit to a remaining candidate.

The implementation maintains:
- phase-local statistics,
- active-arm tracking,
- configurable elimination schedules.

---

## Exp3 and Exp3-IX

Adversarial bandit algorithms based on multiplicative weights updates.

Current implementations include:
- fixed learning-rate Exp3,
- adaptive learning-rate variants,
- Exp3-IX with implicit exploration,
- doubling-trick wrappers.

The implementations focus on:
- probability stability,
- importance-weighted updates,
- reusable update structures.

Since reduction-based no-swap-regret algorithms rely heavily on external learners, Exp3-based learners are expected to become important building blocks for future extensions.

---

# Wrappers

Public constructors are separated from internal implementations.

For example:

```text
ucb.py
ucb_wrappers.py
```

This separation allows:
- cleaner experiment configuration,
- reusable parameter schedules,
- simpler algorithm composition,
- easier future extensions.

The wrapper structure is especially useful for:
- doubling-trick integration,
- adaptive parameter schedules,
- future reduction-based algorithms.

---

# Future Extensions

Planned future additions include:
- Hedge,
- full-information external-regret learners,
- internal-regret algorithms,
- reduction-based no-swap-regret algorithms,
- Blum-Mansour style reductions,
- Ito-style reductions,
- parallelized learner updates.

The long-term goal is to study:
- computational efficiency,
- regret convergence,
- finite-horizon behavior,
- scalability of reduction-based methods.
