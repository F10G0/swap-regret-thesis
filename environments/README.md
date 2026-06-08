# Environments

Repeated-game environments used for evaluating learning algorithms.

## Structure

```text
environments/
├── base.py
│
├── full_information/
│   └── repeated_game.py
│
└── partial_information/
    └── bandit_repeated_game.py
```

## Files

### base.py

Core environment definitions.

Provides:
- environment interfaces,
- outcome representation,
- fixed payoff-tensor environments.

### full_information/repeated_game.py

Full-information repeated game.

Each player observes the payoff of all available actions against the realized actions of the other players.

### partial_information/bandit_repeated_game.py

Bandit-feedback repeated game.

Each player observes only the payoff of the action actually played.
