import csv
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from config import (
    ADVERSARIAL_ACTIONS,
    ADVERSARIAL_MEMORY_WINDOW,
    ADVERSARIAL_RAW_DIR,
    HORIZON,
    SEED,
)
from environments import HistoricalFrequencyAdversary, LazyRandomWalkEnvironment
from environments.adversarial import (
    RANDOM_WALK_INITIALIZATIONS,
    RANDOM_WALK_STEP,
)
from experiments.recorder import CsvRecorder, read_final_csv_rows
from experiments.result_schema import (
    EXPECTED_REGRET_FIELDNAMES,
    REALIZED_REGRET_FIELDNAMES,
)
from experiments.runner import ExperimentCancelled
from experiments.scenarios.bandit_cross_play import ALGORITHMS as BANDIT_ALGORITHMS
from experiments.scenarios.full_information_cross_play import (
    ALGORITHMS as FULL_INFORMATION_ALGORITHMS,
)
from metrics.regret import RegretBundles


HISTORICAL_FREQUENCY_ENVIRONMENT = "historical_frequency_v2"
RANDOM_WALK_ENVIRONMENT = "lazy_random_walk_v1"
ENVIRONMENT_LABELS = {
    HISTORICAL_FREQUENCY_ENVIRONMENT: "Historical-frequency adversary",
    RANDOM_WALK_ENVIRONMENT: "Independent lazy random walk",
}
INITIALIZATION_LABELS = {
    "centered": "Centered at 0.5",
    "uniform_grid": "Uniform over the reward grid",
}
MAX_ADVERSARIAL_ACTIONS = 100
ALGORITHMS_BY_FEEDBACK_MODE = {
    "full_information": FULL_INFORMATION_ALGORITHMS,
    "bandit": BANDIT_ALGORITHMS,
}
FEEDBACK_MODE_LABELS = {
    "full_information": "Full information",
    "bandit": "Bandit feedback",
}
TARGET_REGRET_BY_ALGORITHM = {
    "hedge": "external",
    "exp3": "external",
    "exp3_ix": "external",
    "bm": "swap",
    "ito": "swap",
    "lce_ix": "swap",
    "regret_matching": "internal",
    "stationary_regret_matching": "internal",
}
ADVERSARIAL_RESULT_FIELDNAMES = [
    "run_id",
    "environment",
    "initialization_mode",
    "reward_step",
    "environment_seed",
    "learner_seed",
    "feedback_mode",
    "n_actions",
    "memory",
    "algorithm",
    "horizon",
    "t",
    "action",
    "punished_action",
    "payoff",
    "current_best_action",
    "current_best_reward",
    *EXPECTED_REGRET_FIELDNAMES,
    *REALIZED_REGRET_FIELDNAMES,
]
ADVERSARIAL_IDENTITY_FIELDS = (
    "run_id",
    "environment",
    "initialization_mode",
    "reward_step",
    "environment_seed",
    "learner_seed",
    "feedback_mode",
    "n_actions",
    "memory",
    "algorithm",
    "horizon",
)


def adversarial_memory_window(memory: str) -> int:
    if memory == "full_history":
        return 0
    if memory == "previous_action":
        return 1
    if memory.startswith("last_"):
        try:
            window = int(memory.removeprefix("last_"))
        except ValueError:
            pass
        else:
            if window > 0:
                return window
    raise ValueError(f"unknown adversarial memory: {memory}")


def adversarial_memory_key(memory_window: int) -> str:
    return "full_history" if memory_window == 0 else f"last_{memory_window}"


def adversarial_memory_label(memory_window: int) -> str:
    if memory_window == 0:
        return "Full history"
    return f"Last {memory_window} round" if memory_window == 1 else f"Last {memory_window} rounds"


def adversarial_environment_detail(row: dict[str, str], include_environment_seed: bool = False) -> str:
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        return adversarial_memory_label(adversarial_memory_window(row["memory"]))
    detail = INITIALIZATION_LABELS[row["initialization_mode"]]
    return f"{detail} · environment seed {row['environment_seed']}" if include_environment_seed else detail


@dataclass(frozen=True)
class AdversarialExperimentSpec:
    algorithm_name: str
    n_actions: int
    horizon: int
    seed: int
    memory_window: int = ADVERSARIAL_MEMORY_WINDOW
    feedback_mode: str = "full_information"
    environment: str = HISTORICAL_FREQUENCY_ENVIRONMENT
    initialization_mode: str = "centered"
    environment_seed: int = SEED

    def __post_init__(self) -> None:
        if self.feedback_mode not in ALGORITHMS_BY_FEEDBACK_MODE:
            raise ValueError(f"unknown feedback mode: {self.feedback_mode}")
        if self.algorithm_name not in ALGORITHMS_BY_FEEDBACK_MODE[self.feedback_mode]:
            raise ValueError(
                f"algorithm {self.algorithm_name} is not available for "
                f"{self.feedback_mode}"
            )
        if not 2 <= self.n_actions <= MAX_ADVERSARIAL_ACTIONS:
            raise ValueError(f"number of actions must be between 2 and {MAX_ADVERSARIAL_ACTIONS}")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.environment not in ENVIRONMENT_LABELS:
            raise ValueError(f"unknown adversarial environment: {self.environment}")
        if self.environment == HISTORICAL_FREQUENCY_ENVIRONMENT:
            if self.memory_window < 0:
                raise ValueError("memory window must be non-negative")
        else:
            if self.initialization_mode not in RANDOM_WALK_INITIALIZATIONS:
                raise ValueError(f"unknown random-walk initialization: {self.initialization_mode}")
            if self.environment_seed < 0:
                raise ValueError("environment seed must be non-negative")

    def configuration(self) -> dict:
        random_walk = self.environment == RANDOM_WALK_ENVIRONMENT
        return {
            "environment": self.environment,
            "initialization_mode": self.initialization_mode if random_walk else "",
            "reward_step": RANDOM_WALK_STEP if random_walk else "",
            "environment_seed": self.environment_seed if random_walk else "",
            "learner_seed": self.seed,
            "feedback_mode": self.feedback_mode,
            "n_actions": self.n_actions,
            "memory": "" if random_walk else adversarial_memory_key(self.memory_window),
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
        }

    @property
    def run_id(self) -> str:
        if self.environment == HISTORICAL_FREQUENCY_ENVIRONMENT:
            identity = {
                "environment": self.environment,
                "n_actions": self.n_actions,
                "memory": adversarial_memory_key(self.memory_window),
                "algorithm": self.algorithm_name,
                "horizon": self.horizon,
                "seed": self.seed,
            }
            if self.feedback_mode != "full_information":
                identity["feedback_mode"] = self.feedback_mode
            prefix = "historical_frequency"
        else:
            identity = self.configuration()
            prefix = "lazy_random_walk"
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{self.n_actions}a_{self.algorithm_name}_{digest}"


def run_adversarial_experiment(
    algorithm_name: str,
    n_actions: int = ADVERSARIAL_ACTIONS,
    horizon: int = HORIZON,
    seed: int = SEED,
    memory_window: int = ADVERSARIAL_MEMORY_WINDOW,
    output_dir: str | Path = ADVERSARIAL_RAW_DIR,
    should_cancel: Callable[[], bool] | None = None,
    feedback_mode: str = "full_information",
    environment: str = HISTORICAL_FREQUENCY_ENVIRONMENT,
    initialization_mode: str = "centered",
    environment_seed: int = SEED,
) -> Path:
    spec = AdversarialExperimentSpec(
        algorithm_name=algorithm_name,
        n_actions=n_actions,
        horizon=horizon,
        seed=seed,
        memory_window=memory_window,
        feedback_mode=feedback_mode,
        environment=environment,
        initialization_mode=initialization_mode,
        environment_seed=environment_seed,
    )
    output_path = Path(output_dir) / f"{spec.run_id}.csv"
    if output_path.exists():
        raise FileExistsError(
            f"adversarial experiment {spec.run_id} already exists at {output_path}"
        )

    historical = spec.environment == HISTORICAL_FREQUENCY_ENVIRONMENT
    if historical:
        experiment_environment = HistoricalFrequencyAdversary(
            spec.n_actions,
            memory_window=spec.memory_window or None,
        )
    else:
        experiment_environment = LazyRandomWalkEnvironment(
            spec.n_actions,
            spec.horizon,
            spec.environment_seed,
            spec.initialization_mode,
        )
    learner = ALGORITHMS_BY_FEEDBACK_MODE[spec.feedback_mode][spec.algorithm_name].create(
        spec.n_actions,
        spec.horizon,
        spec.seed,
    )
    regrets = RegretBundles(spec.n_actions)
    metadata = spec.configuration() | {"run_id": spec.run_id}

    with CsvRecorder(ADVERSARIAL_RESULT_FIELDNAMES, output_path) as recorder:
        for time in range(1, spec.horizon + 1):
            if should_cancel is not None and should_cancel():
                raise ExperimentCancelled("experiment cancelled")

            strategy = learner.strategy()
            action = learner.sample_action()
            if historical:
                experiment_environment.step((action,))
                punished_action = experiment_environment.punished_action
            else:
                experiment_environment.step()
                punished_action = ""
            payoffs = experiment_environment.feedback()
            regrets.update(strategy, action, payoffs)
            feedback = payoffs if spec.feedback_mode == "full_information" else float(payoffs[action])
            learner.update(feedback)

            recorder.record(
                {
                    **metadata,
                    "t": time,
                    "action": action,
                    "punished_action": punished_action,
                    "payoff": float(payoffs[action]),
                    "current_best_action": int(np.argmax(payoffs)),
                    "current_best_reward": float(np.max(payoffs)),
                    **regrets.expected.summary(time),
                    **regrets.realized.summary(time),
                }
            )

    return output_path


def _validate_adversarial_row(row: dict[str, str], input_path: Path) -> int:
    if row["feedback_mode"] not in ALGORITHMS_BY_FEEDBACK_MODE:
        raise ValueError(f"{input_path} contains an invalid feedback mode")
    if row["algorithm"] not in ALGORITHMS_BY_FEEDBACK_MODE[row["feedback_mode"]]:
        raise ValueError(f"{input_path} contains an invalid algorithm")
    if row["environment"] not in ENVIRONMENT_LABELS:
        raise ValueError(f"{input_path} contains an invalid environment")
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        adversarial_memory_window(row["memory"])
    else:
        if row["initialization_mode"] not in RANDOM_WALK_INITIALIZATIONS:
            raise ValueError(f"{input_path} contains an invalid initialization")
        if not np.isclose(float(row["reward_step"]), RANDOM_WALK_STEP):
            raise ValueError(f"{input_path} contains an invalid reward step")
        if int(row["environment_seed"]) < 0:
            raise ValueError(f"{input_path} contains an invalid environment seed")
    if int(row["learner_seed"]) < 0:
        raise ValueError(f"{input_path} contains an invalid learner seed")

    horizon = int(row["horizon"])
    n_actions = int(row["n_actions"])
    if horizon <= 0:
        raise ValueError(f"{input_path} contains invalid round metadata")
    if not 0 <= int(row["action"]) < n_actions:
        raise ValueError(f"{input_path} contains an invalid action")
    if not 0 <= int(row["current_best_action"]) < n_actions:
        raise ValueError(f"{input_path} contains an invalid best action")
    if not 0.0 <= float(row["current_best_reward"]) <= 1.0:
        raise ValueError(f"{input_path} contains an invalid best reward")
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        if not 0 <= int(row["punished_action"]) < n_actions:
            raise ValueError(f"{input_path} contains an invalid punished action")
    return horizon


def load_adversarial_rows(
    input_path: str | Path,
    max_points: int | None = None,
) -> list[dict[str, str]]:
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive")
    input_path = Path(input_path)
    rows = []
    expected_identity = None
    expected_time = 1
    stride = 1
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(ADVERSARIAL_RESULT_FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"{input_path} is missing required columns: {columns}")
        for row in reader:
            horizon = _validate_adversarial_row(row, input_path)
            identity = tuple(row[field] for field in ADVERSARIAL_IDENTITY_FIELDS)
            if expected_identity is None:
                expected_identity = identity
                if max_points is not None:
                    stride = max(1, (horizon + max_points - 1) // max_points)
            elif identity != expected_identity:
                raise ValueError(f"{input_path} contains inconsistent metadata")

            time = int(row["t"])
            if time != expected_time or time > horizon:
                raise ValueError(f"{input_path} contains invalid round metadata")

            if time == 1 or time == horizon or time % stride == 0:
                rows.append(row)
            expected_time += 1

    if expected_identity is None:
        raise ValueError(f"{input_path} is empty")
    if expected_time - 1 != int(rows[-1]["horizon"]):
        raise ValueError(f"{input_path} does not contain the complete trajectory")
    return rows


def load_final_adversarial_row(input_path: str | Path) -> dict[str, str]:
    input_path = Path(input_path)
    fieldnames, rows = read_final_csv_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    missing = set(ADVERSARIAL_RESULT_FIELDNAMES) - fieldnames
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{input_path} is missing required columns: {columns}")
    row = rows[0]
    horizon = _validate_adversarial_row(row, input_path)
    if int(row["t"]) != horizon:
        raise ValueError(f"{input_path} has no complete final round")
    return row
