import csv
from collections.abc import Callable
from dataclasses import dataclass, field
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
    RANDOM_WALK_STEP,
)
from experiments.recorder import CsvRecorder, read_final_csv_rows, require_csv_columns
from experiments.recording import MAX_RECORDED_POINTS, recording_checkpoints
from experiments.sampling import CheckpointRows
from experiments.result_schema import (
    REGRET_FIELDNAMES,
)
from experiments.runtime_environment import (
    runtime_environment_fingerprint,
    runtime_environment_json,
    validate_runtime_environment,
)
from experiments.runner import ExperimentCancelled
from experiments.seeding import (
    ENVIRONMENT_SEED_DOMAIN,
    LEARNER_SEED_DOMAIN,
    domain_separated_seed,
)
from experiments.scenarios.cross_play import ALGORITHMS_BY_FEEDBACK_MODE
from metrics.regret import RegretBundle


HISTORICAL_FREQUENCY_ENVIRONMENT = "historical_frequency_v3"
RANDOM_WALK_ENVIRONMENT = "lazy_random_walk_v1"
ENVIRONMENT_LABELS = {
    HISTORICAL_FREQUENCY_ENVIRONMENT: "Historical-frequency adversary",
    RANDOM_WALK_ENVIRONMENT: "Independent lazy random walk",
}
MAX_ADVERSARIAL_ACTIONS = 100
TARGET_REGRET_BY_ALGORITHM = {
    "hedge": "external",
    "auer_exp3": "external",
    "exp3_ix": "external",
    "bm": "swap",
    "ito": "swap",
    "lce_ix": "swap",
    "regret_matching": "internal",
    "stationary_regret_matching": "internal",
}
ADVERSARIAL_IDENTITY_FIELDS = (
    "run_id",
    "environment",
    "reward_step",
    "base_environment_seed",
    "environment_seed",
    "base_learner_seed",
    "learner_seed",
    "replicate",
    "runtime_environment",
    "runtime_fingerprint",
    "feedback_mode",
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


def adversarial_result_fieldnames() -> list[str]:
    return ADVERSARIAL_BASE_FIELDNAMES + REGRET_FIELDNAMES


def adversarial_environment_detail(row: dict[str, str]) -> str:
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        return "Full history · top half punished"
    return "Centered at 0.5"


@dataclass(frozen=True)
class AdversarialExperimentSpec:
    algorithm_name: str
    n_actions: int
    horizon: int
    seed: int
    feedback_mode: str = "full_information"
    environment: str = HISTORICAL_FREQUENCY_ENVIRONMENT
    replicate: int = 0
    runtime_environment: str = field(default_factory=runtime_environment_json)

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
        canonical_runtime = validate_runtime_environment(self.runtime_environment)
        object.__setattr__(self, "runtime_environment", canonical_runtime)
        if self.environment not in ENVIRONMENT_LABELS:
            raise ValueError(f"unknown adversarial environment: {self.environment}")
    def configuration(self) -> dict:
        random_walk = self.environment == RANDOM_WALK_ENVIRONMENT
        configuration = {
            "environment": self.environment,
            "reward_step": RANDOM_WALK_STEP if random_walk else "",
            "base_environment_seed": self.seed if random_walk else "",
            "environment_seed": self.replicate_environment_seed if random_walk else "",
            "base_learner_seed": self.seed,
            "learner_seed": self.learner_seed,
            "replicate": self.replicate,
            "feedback_mode": self.feedback_mode,
            "n_actions": self.n_actions,
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
        }
        configuration["runtime_environment"] = self.runtime_environment
        configuration["runtime_fingerprint"] = self.runtime_fingerprint
        return configuration

    @property
    def learner_seed(self) -> int:
        return domain_separated_seed(
            self.seed,
            self.replicate,
            LEARNER_SEED_DOMAIN,
        )

    @property
    def replicate_environment_seed(self) -> int:
        return domain_separated_seed(
            self.seed,
            self.replicate,
            ENVIRONMENT_SEED_DOMAIN,
        )

    @property
    def runtime_fingerprint(self) -> str:
        return runtime_environment_fingerprint(self.runtime_environment)

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
            identity["runtime_fingerprint"] = self.runtime_fingerprint
            if self.replicate:
                identity["replicate"] = self.replicate
            if self.feedback_mode != "full_information":
                identity["feedback_mode"] = self.feedback_mode
            prefix = "historical_frequency"
        else:
            identity = self.configuration()
            if not self.replicate:
                identity.pop("replicate")
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
    replicate: int = 0,
    runtime_environment: str | None = None,
    max_recorded_points: int = MAX_RECORDED_POINTS,
) -> Path:
    spec = AdversarialExperimentSpec(
        algorithm_name=algorithm_name,
        n_actions=n_actions,
        horizon=horizon,
        seed=seed,
        feedback_mode=feedback_mode,
        environment=environment,
        replicate=replicate,
        runtime_environment=(
            runtime_environment_json()
            if runtime_environment is None
            else runtime_environment
        ),
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
        )
    learner = ALGORITHMS_BY_FEEDBACK_MODE[spec.feedback_mode][spec.algorithm_name].create(
        spec.n_actions,
        spec.horizon,
        spec.learner_seed,
    )
    regrets = RegretBundle(spec.n_actions)
    checkpoints = set(recording_checkpoints(spec.horizon, max_recorded_points))
    metadata = spec.configuration() | {"run_id": spec.run_id}

    with CsvRecorder(
        adversarial_result_fieldnames(),
        output_path,
    ) as recorder:
        for time in range(1, spec.horizon + 1):
            if should_cancel is not None and should_cancel():
                raise ExperimentCancelled("experiment cancelled")

            strategy = learner.strategy()
            action = learner.sample_action()
            if historical:
                experiment_environment.step((action,))
            else:
                experiment_environment.step()
            payoffs = experiment_environment.feedback()
            regrets.update(strategy, payoffs)
            feedback = payoffs if spec.feedback_mode == "full_information" else float(payoffs[action])
            learner.update(feedback)

            if time not in checkpoints:
                continue
            punished_actions = " ".join(map(str, experiment_environment.punished_actions)) if historical else ""
            regret_summary = regrets.summary(time)
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


def _validate_adversarial_metadata(row: dict[str, str], input_path: Path) -> int:
    if row["feedback_mode"] not in ALGORITHMS_BY_FEEDBACK_MODE:
        raise ValueError(f"{input_path} contains an invalid feedback mode")
    if row["algorithm"] not in ALGORITHMS_BY_FEEDBACK_MODE[row["feedback_mode"]]:
        raise ValueError(f"{input_path} contains an invalid algorithm")
    if row["environment"] not in ENVIRONMENT_LABELS:
        raise ValueError(f"{input_path} contains an invalid environment")
    if row["environment"] == RANDOM_WALK_ENVIRONMENT:
        if not np.isclose(float(row["reward_step"]), RANDOM_WALK_STEP):
            raise ValueError(f"{input_path} contains an invalid reward step")
        if int(row["base_environment_seed"]) < 0 or int(row["environment_seed"]) < 0:
            raise ValueError(f"{input_path} contains an invalid environment seed")
    if int(row["base_learner_seed"]) < 0 or int(row["learner_seed"]) < 0:
        raise ValueError(f"{input_path} contains an invalid learner seed")
    if int(row["replicate"]) < 0:
        raise ValueError(f"{input_path} contains an invalid replicate")
    replicate = int(row["replicate"])
    expected_learner_seed = domain_separated_seed(
        int(row["base_learner_seed"]),
        replicate,
        LEARNER_SEED_DOMAIN,
    )
    if int(row["learner_seed"]) != expected_learner_seed:
        raise ValueError(f"{input_path} contains an invalid learner seed schedule")
    if row["environment"] == RANDOM_WALK_ENVIRONMENT:
        expected_environment_seed = domain_separated_seed(
            int(row["base_environment_seed"]),
            replicate,
            ENVIRONMENT_SEED_DOMAIN,
        )
        if int(row["environment_seed"]) != expected_environment_seed:
            raise ValueError(f"{input_path} contains an invalid environment seed schedule")
    runtime = validate_runtime_environment(row["runtime_environment"])
    if runtime_environment_fingerprint(runtime) != row["runtime_fingerprint"]:
        raise ValueError(f"{input_path} contains an invalid runtime fingerprint")

    horizon = int(row["horizon"])
    if horizon <= 0:
        raise ValueError(f"{input_path} contains invalid round metadata")
    return horizon


def _validate_adversarial_observation(row: dict[str, str], input_path: Path, n_actions: int) -> None:
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


def _validate_adversarial_row(row: dict[str, str], input_path: Path) -> int:
    horizon = _validate_adversarial_metadata(row, input_path)
    _validate_adversarial_observation(row, input_path, int(row["n_actions"]))
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
    previous_time = 0
    sampler = None
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        for row in reader:
            # Compare the original metadata on EVERY row, including rows omitted
            # from plots. Seed derivation and runtime JSON/hashing need run only once.
            identity = tuple(row.get(field) for field in ADVERSARIAL_IDENTITY_FIELDS)
            if expected_identity is None:
                require_csv_columns(input_path, fieldnames, set(adversarial_result_fieldnames()))
                horizon = _validate_adversarial_metadata(row, input_path)
                n_actions = int(row["n_actions"])
                expected_identity = identity
                if max_points is not None:
                    sampler = CheckpointRows(horizon, max_points)
            elif identity != expected_identity:
                raise ValueError(f"{input_path} contains inconsistent metadata")

            _validate_adversarial_observation(row, input_path, n_actions)

            time = int(row["t"])
            if time <= previous_time or time > horizon or (previous_time == 0 and time != 1):
                raise ValueError(f"{input_path} contains invalid round metadata")

            if sampler is None:
                rows.append(row)
            else:
                sampler.add(row)
            previous_time = time

    if expected_identity is None:
        raise ValueError(f"{input_path} is empty")
    if sampler is not None:
        rows = sampler.rows()
    if previous_time != int(rows[-1]["horizon"]):
        raise ValueError(f"{input_path} does not contain the complete trajectory")
    return rows


def load_final_adversarial_row(input_path: str | Path) -> dict[str, str]:
    input_path = Path(input_path)
    fieldnames, rows = read_final_csv_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    row = rows[0]
    require_csv_columns(input_path, fieldnames, set(adversarial_result_fieldnames()))
    horizon = _validate_adversarial_row(row, input_path)
    if int(row["t"]) != horizon:
        raise ValueError(f"{input_path} has no complete final round")
    return row
