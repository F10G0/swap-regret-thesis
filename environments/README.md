# Environments Module

This module contains reward-generating environments used for empirical evaluation of learning algorithms.

Current environments include:
- stochastic Bernoulli bandits,
- adversarial bandits with predefined reward sequences.

All environments follow a unified interface supporting:
- reward generation,
- optimal action queries,
- regret computation.

The current focus is on:
- stochastic bandit settings,
- adversarial MAB settings.

Future extensions may include:
- full-information environments,
- game environments,
- correlated equilibrium simulations.
