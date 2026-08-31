import csv
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from config import (
    ADVERSARIAL_ACTIONS,
    ADVERSARIAL_RAW_DIR,
    HORIZON,
    SEED,
)
from environments import HistoricalFrequencyAdversary, LazyRandomWalkEnvironment
from environments.adversarial import (
    RANDOM_WALK_INITIALIZATIONS,
    RANDOM_WALK_STEP,
)
from experiments.recorder import CsvRecorder, read_final_csv_rows, require_csv_columns
from experiments.result_schema import (
    REGRET_FIELDNAMES,
    RESULT_IMPLEMENTATION_VERSION,
    regret_sources,
    resolve_regret_evaluation,
)
from experiments.results import result_regret_evaluation
from experiments.runner import ExperimentCancelled
from experiments.scenarios.bandit_cross_play import ALGORITHMS as BANDIT_ALGORITHMS
from experiments.scenarios.full_information_cross_play import (
    ALGORITHMS as FULL_INFORMATION_ALGORITHMS,
)
from metrics.regret import RegretBundles


HISTORICAL_FREQUENCY_ENVIRONMENT = "historical_frequency_v3"
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
ADVERSARIAL_IDENTITY_FIELDS = (
    "run_id",
    "implementation_version",
    "environment",
    "initialization_mode",
    "reward_step",
    "environment_seed",
    "learner_seed",
    "replicate",
    "feedback_mode",
    "regret_evaluation",
    "n_actions",
    "algorithm",
    "horizon",
)
ADVERSARIAL_BASE_FIELDNAMES = [
    *ADVERSARIAL_IDENTITY_FIELDS,
    "t",
    "action",
    "punished_actions",
    "payoff",
    "current_best_action",
    "current_best_reward",
]
ADVERSARIAL_LEGACY_FIELDS = {"replicate", "regret_evaluation", "implementation_version"}


def adversarial_result_fieldnames(regret_evaluation: str) -> list[str]:
    return ADVERSARIAL_BASE_FIELDNAMES + [
        field
        for source in regret_sources(regret_evaluation)
        for field in REGRET_FIELDNAMES[source]
    ]


def _normalize_adversarial_row(row: dict[str, str]) -> None:
    row.setdefault("replicate", "0")
    row.setdefault("implementation_version", "0")
    row["regret_evaluation"] = result_regret_evaluation(row)


def adversarial_environment_detail(row: dict[str, str]) -> str:
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        return "Full history · top half punished"
    return INITIALIZATION_LABELS[row["initialization_mode"]]


@dataclass(frozen=True)
class AdversarialExperimentSpec:
    algorithm_name: str
    n_actions: int
    horizon: int
    seed: int
    feedback_mode: str = "full_information"
    environment: str = HISTORICAL_FREQUENCY_ENVIRONMENT
    initialization_mode: str = "centered"
    environment_seed: int = SEED
    replicate: int = 0
    regret_evaluation: str = "both"
    implementation_version: int = RESULT_IMPLEMENTATION_VERSION

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
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        if self.implementation_version < 0:
            raise ValueError("implementation_version must be non-negative")
        object.__setattr__(
            self,
            "regret_evaluation",
            resolve_regret_evaluation(self.feedback_mode, self.regret_evaluation),
        )
        if self.environment not in ENVIRONMENT_LABELS:
            raise ValueError(f"unknown adversarial environment: {self.environment}")
        if self.environment == RANDOM_WALK_ENVIRONMENT:
            if self.initialization_mode not in RANDOM_WALK_INITIALIZATIONS:
                raise ValueError(f"unknown random-walk initialization: {self.initialization_mode}")
            if self.environment_seed < 0:
                raise ValueError("environment seed must be non-negative")

    def configuration(self) -> dict:
        random_walk = self.environment == RANDOM_WALK_ENVIRONMENT
        configuration = {
            "environment": self.environment,
            "initialization_mode": self.initialization_mode if random_walk else "",
            "reward_step": RANDOM_WALK_STEP if random_walk else "",
            "environment_seed": self.replicate_environment_seed if random_walk else "",
            "learner_seed": self.learner_seed,
            "replicate": self.replicate,
            "feedback_mode": self.feedback_mode,
            "regret_evaluation": self.regret_evaluation,
            "n_actions": self.n_actions,
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
        }
        if self.implementation_version:
            configuration["implementation_version"] = self.implementation_version
        return configuration

    @property
    def learner_seed(self) -> int:
        return self.seed + self.replicate

    @property
    def replicate_environment_seed(self) -> int:
        return self.environment_seed + self.replicate

    @property
    def run_id(self) -> str:
        if self.environment == HISTORICAL_FREQUENCY_ENVIRONMENT:
            identity = {
                "environment": self.environment,
                "n_actions": self.n_actions,
                "algorithm": self.algorithm_name,
                "horizon": self.horizon,
                "seed": self.seed,
            }
            if self.implementation_version:
                identity["implementation_version"] = self.implementation_version
            if self.replicate:
                identity["replicate"] = self.replicate
            if self.feedback_mode != "full_information":
                identity["feedback_mode"] = self.feedback_mode
            if self.regret_evaluation != "both":
                identity["regret_evaluation"] = self.regret_evaluation
            prefix = "historical_frequency"
        else:
            identity = self.configuration()
            if not self.replicate:
                identity.pop("replicate")
            if self.regret_evaluation == "both":
                identity.pop("regret_evaluation")
            prefix = "lazy_random_walk"
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{self.n_actions}a_{self.algorithm_name}_{digest}"


def run_adversarial_experiment(
    algorithm_name: str,
    n_actions: int = ADVERSARIAL_ACTIONS,
    horizon: int = HORIZON,
    seed: int = SEED,
    output_dir: str | Path = ADVERSARIAL_RAW_DIR,
    should_cancel: Callable[[], bool] | None = None,
    feedback_mode: str = "full_information",
    environment: str = HISTORICAL_FREQUENCY_ENVIRONMENT,
    initialization_mode: str = "centered",
    environment_seed: int = SEED,
    replicate: int = 0,
    regret_evaluation: str = "both",
    implementation_version: int = RESULT_IMPLEMENTATION_VERSION,
) -> Path:
    spec = AdversarialExperimentSpec(
        algorithm_name=algorithm_name,
        n_actions=n_actions,
        horizon=horizon,
        seed=seed,
        feedback_mode=feedback_mode,
        environment=environment,
        initialization_mode=initialization_mode,
        environment_seed=environment_seed,
        replicate=replicate,
        regret_evaluation=regret_evaluation,
        implementation_version=implementation_version,
    )
    output_path = Path(output_dir) / f"{spec.run_id}.csv"
    if output_path.exists():
        raise FileExistsError(
            f"adversarial experiment {spec.run_id} already exists at {output_path}"
        )

    historical = spec.environment == HISTORICAL_FREQUENCY_ENVIRONMENT
    if historical:
        experiment_environment = HistoricalFrequencyAdversary(spec.n_actions)
    else:
        experiment_environment = LazyRandomWalkEnvironment(
            spec.n_actions,
            spec.horizon,
            spec.replicate_environment_seed,
            spec.initialization_mode,
        )
    learner = ALGORITHMS_BY_FEEDBACK_MODE[spec.feedback_mode][spec.algorithm_name].create(
        spec.n_actions,
        spec.horizon,
        spec.learner_seed,
    )
    regrets = RegretBundles(spec.n_actions)
    metadata = spec.configuration() | {"run_id": spec.run_id}

    with CsvRecorder(
        adversarial_result_fieldnames(spec.regret_evaluation),
        output_path,
    ) as recorder:
        for time in range(1, spec.horizon + 1):
            if should_cancel is not None and should_cancel():
                raise ExperimentCancelled("experiment cancelled")

            strategy = learner.strategy()
            action = learner.sample_action()
            if historical:
                experiment_environment.step((action,))
                punished_actions = " ".join(map(str, experiment_environment.punished_actions))
            else:
                experiment_environment.step()
                punished_actions = ""
            payoffs = experiment_environment.feedback()
            regrets.update(strategy, action, payoffs)
            feedback = payoffs if spec.feedback_mode == "full_information" else float(payoffs[action])
            learner.update(feedback)

            regret_summary = {}
            for source in regret_sources(spec.regret_evaluation):
                regret_summary.update(getattr(regrets, source).summary(time))
            recorder.record(
                {
                    **metadata,
                    "t": time,
                    "action": action,
                    "punished_actions": punished_actions,
                    "payoff": float(payoffs[action]),
                    "current_best_action": int(np.argmax(payoffs)),
                    "current_best_reward": float(np.max(payoffs)),
                    **regret_summary,
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
    if row["environment"] == RANDOM_WALK_ENVIRONMENT:
        if row["initialization_mode"] not in RANDOM_WALK_INITIALIZATIONS:
            raise ValueError(f"{input_path} contains an invalid initialization")
        if not np.isclose(float(row["reward_step"]), RANDOM_WALK_STEP):
            raise ValueError(f"{input_path} contains an invalid reward step")
        if int(row["environment_seed"]) < 0:
            raise ValueError(f"{input_path} contains an invalid environment seed")
    if int(row["learner_seed"]) < 0:
        raise ValueError(f"{input_path} contains an invalid learner seed")
    if int(row["replicate"]) < 0:
        raise ValueError(f"{input_path} contains an invalid replicate")
    if int(row["implementation_version"]) < 0:
        raise ValueError(f"{input_path} contains an invalid implementation version")

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
        try:
            punished_actions = tuple(map(int, row["punished_actions"].split()))
        except ValueError as error:
            raise ValueError(f"{input_path} contains invalid punished actions") from error
        if len(punished_actions) != (n_actions + 1) // 2 or len(set(punished_actions)) != len(punished_actions) or any(not 0 <= action < n_actions for action in punished_actions):
            raise ValueError(f"{input_path} contains invalid punished actions")
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
        fieldnames = set(reader.fieldnames or ())
        require_csv_columns(input_path, fieldnames, set(ADVERSARIAL_BASE_FIELDNAMES) - ADVERSARIAL_LEGACY_FIELDS)
        for row in reader:
            _normalize_adversarial_row(row)
            horizon = _validate_adversarial_row(row, input_path)
            identity = tuple(row[field] for field in ADVERSARIAL_IDENTITY_FIELDS)
            if expected_identity is None:
                required = set(adversarial_result_fieldnames(row["regret_evaluation"])) - ADVERSARIAL_LEGACY_FIELDS
                require_csv_columns(input_path, fieldnames, required)
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
    require_csv_columns(input_path, fieldnames, set(ADVERSARIAL_BASE_FIELDNAMES) - ADVERSARIAL_LEGACY_FIELDS)
    row = rows[0]
    _normalize_adversarial_row(row)
    required = set(adversarial_result_fieldnames(row["regret_evaluation"])) - ADVERSARIAL_LEGACY_FIELDS
    require_csv_columns(input_path, fieldnames, required)
    horizon = _validate_adversarial_row(row, input_path)
    if int(row["t"]) != horizon:
        raise ValueError(f"{input_path} has no complete final round")
    return row
