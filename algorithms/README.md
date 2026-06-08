# Algorithms

Implementation of regret-minimization algorithms used in this project.

## Structure

```text
algorithms/
├── base.py
│
├── external_regret/
│   ├── base.py
│   ├── full_information/
│   │   └── hedge.py
│   └── partial_information/
│       └── exp3.py
│
└── swap_regret/
    ├── base.py
    ├── stationary.py
    ├── full_information/
    │   ├── blum_mansour.py
    │   └── ito.py
    └── partial_information/
        ├── blum_mansour.py
        └── ito.py
```

## Files

### base.py

Base class for all learning algorithms.

Provides:
- strategy management,
- action sampling,
- common validation utilities,
- unified update/reset interface.

### external_regret/

External-regret minimization algorithms.

#### base.py

Shared functionality for exponential-weights methods.

#### full_information/hedge.py

Implementation of the Hedge algorithm.

#### partial_information/exp3.py

Implementation of the Exp3 algorithm.

### swap_regret/

Swap-regret minimization algorithms based on stationary-distribution reductions.

#### base.py

Base implementation for stationary-distribution reductions.

#### stationary.py

Utilities for computing stationary distributions.

#### full_information/

Full-information versions of swap-regret reductions.

- blum_mansour.py
- ito.py

#### partial_information/

Bandit-feedback versions of swap-regret reductions.

- blum_mansour.py
- ito.py
