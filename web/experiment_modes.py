from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from experiments.scenarios.bandit_cross_play import ALGORITHMS as BANDIT_ALGORITHMS, run_bandit_cross_play_experiment
from experiments.scenarios.cross_play import AlgorithmFactory
from experiments.scenarios.full_information_cross_play import (
    ALGORITHMS as FULL_INFORMATION_ALGORITHMS,
    run_full_information_cross_play_experiment,
)


@dataclass(frozen=True)
class FeedbackMode:
    label: str
    algorithms: dict[str, AlgorithmFactory]
    runner: Callable[..., Path]


FEEDBACK_MODES = {
    "full_information": FeedbackMode(
        label="Full information",
        algorithms=FULL_INFORMATION_ALGORITHMS,
        runner=run_full_information_cross_play_experiment,
    ),
    "bandit": FeedbackMode(
        label="Bandit feedback",
        algorithms=BANDIT_ALGORITHMS,
        runner=run_bandit_cross_play_experiment,
    ),
}

REGRET_EVALUATION_LABELS = {
    "expected": "Expected",
    "realized": "Realized",
    "both": "Both",
}
