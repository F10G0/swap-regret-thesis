import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np

from experiments.recorder import CsvRecorder, require_csv_columns
from experiments.parallel import run_replicates
from experiments.result_schema import (
    REGRET_FIELDNAMES,
    RESULT_IMPLEMENTATION_VERSION,
    result_implementation_version,
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
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
    TARGET_REGRET_BY_ALGORITHM,
    adversarial_environment_detail,
    load_final_adversarial_row,
    run_adversarial_experiment,
)


ACTION_SCALING_IDENTITY_FIELDS = (
    "run_id",
    "implementation_version",
    "runtime_environment",
    "runtime_fingerprint",
    "environment",
    "feedback_mode",
    "algorithm",
    "horizon",
    "base_environment_seed",
    "base_learner_seed",
    "action_counts",
    "replicates",
)
ACTION_SCALING_FIELDNAMES = [
    *ACTION_SCALING_IDENTITY_FIELDS,
    "n_actions",
    "replicate",
    "environment_seed",
    "learner_seed",
    "target_regret",
    *REGRET_FIELDNAMES,
]


@dataclass(frozen=True)
class AdversarialScalingSpec:
    environment: str
    feedback_mode: str
    algorithm_name: str
    action_counts: tuple[int, ...]
    replicates: int
    horizon: int
    environment_seed: int
    learner_seed: int
    implementation_version: int = RESULT_IMPLEMENTATION_VERSION
    runtime_environment: str = field(default_factory=runtime_environment_json)

    def __post_init__(self) -> None:
        action_counts = tuple(sorted(self.action_counts))
        if len(action_counts) < 2:
            raise ValueError("provide at least two action counts")
        if len(set(action_counts)) != len(action_counts):
            raise ValueError("action counts must be unique")
        if self.replicates <= 0:
            raise ValueError("replicates must be positive")
        if self.implementation_version < 0:
            raise ValueError("implementation_version must be non-negative")
        canonical_runtime = validate_runtime_environment(
            self.runtime_environment,
            allow_empty=self.implementation_version == 0,
        )
        object.__setattr__(self, "runtime_environment", canonical_runtime)
        object.__setattr__(self, "action_counts", action_counts)
        for n_actions in action_counts:
            AdversarialExperimentSpec(
                environment=self.environment,
                environment_seed=self.environment_seed,
                feedback_mode=self.feedback_mode,
                algorithm_name=self.algorithm_name,
                n_actions=n_actions,
                horizon=self.horizon,
                seed=self.learner_seed,
                implementation_version=self.implementation_version,
                runtime_environment=self.runtime_environment,
            )

    def configuration(self) -> dict:
        random_walk = self.environment == RANDOM_WALK_ENVIRONMENT
        configuration = {
            "environment": self.environment,
            "feedback_mode": self.feedback_mode,
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
            "base_environment_seed": self.environment_seed if random_walk else "",
            "base_learner_seed": self.learner_seed,
            "action_counts": ",".join(map(str, self.action_counts)),
            "replicates": self.replicates,
        }
        if self.runtime_environment:
            configuration["runtime_environment"] = self.runtime_environment
            configuration["runtime_fingerprint"] = self.runtime_fingerprint
        if self.implementation_version:
            configuration["implementation_version"] = self.implementation_version
        return configuration

    @property
    def runtime_fingerprint(self) -> str:
        if not self.runtime_environment:
            return ""
        return runtime_environment_fingerprint(self.runtime_environment)

    @property
    def run_id(self) -> str:
        identity = self.configuration()
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return f"action_scaling_{self.algorithm_name}_{sha256(payload.encode()).hexdigest()[:10]}"


def adversarial_scaling_environment_detail(row: dict[str, str]) -> str:
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        return adversarial_environment_detail(row)
    return (
        "Centered at 0.5 · "
        f"base environment seed {row['base_environment_seed']}"
    )


def run_adversarial_scaling_experiment(
    spec: AdversarialScalingSpec,
    output_dir: str | Path,
    should_cancel: Callable[[], bool] | None = None,
    completed: Callable[[], None] | None = None,
    workers: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_path = output_dir / f"{spec.run_id}.csv"
    if output_path.exists():
        raise FileExistsError(f"action-space scaling experiment {spec.run_id} already exists")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"run_id": spec.run_id, **spec.configuration()}
    target_regret = TARGET_REGRET_BY_ALGORITHM[spec.algorithm_name]
    with tempfile.TemporaryDirectory(
        prefix=".action-scaling-",
        dir=output_dir,
    ) as temporary_directory:
        tasks = [
            dict(
                environment=spec.environment,
                environment_seed=spec.environment_seed,
                feedback_mode=spec.feedback_mode,
                algorithm_name=spec.algorithm_name,
                n_actions=n_actions,
                horizon=spec.horizon,
                seed=spec.learner_seed,
                replicate=replicate,
                implementation_version=spec.implementation_version,
                runtime_environment=spec.runtime_environment,
                output_dir=temporary_directory,
                max_recorded_points=2,  # Scaling consumes only the final summaries.
            )
            for n_actions in spec.action_counts
            for replicate in range(spec.replicates)
        ]
        paths = run_replicates(
            run_adversarial_experiment, tasks, workers=workers,
            should_cancel=should_cancel, completed=completed,
        )
        with CsvRecorder(ACTION_SCALING_FIELDNAMES, output_path) as recorder:
            for task, result_path in zip(tasks, paths):
                if should_cancel is not None and should_cancel():
                    raise ExperimentCancelled("experiment cancelled")
                final = load_final_adversarial_row(result_path)
                recorder.record({
                    **metadata,
                    "n_actions": task["n_actions"],
                    "replicate": task["replicate"],
                    "environment_seed": final["environment_seed"],
                    "learner_seed": final["learner_seed"],
                    "target_regret": target_regret,
                    **{name: final[name] for name in REGRET_FIELDNAMES},
                })
    return output_path


def load_adversarial_scaling_rows(input_path: str | Path) -> list[dict[str, str]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or ()
        rows = list(reader)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    result_implementation_version(rows[0])
    require_csv_columns(input_path, fieldnames, set(ACTION_SCALING_FIELDNAMES))

    identity = tuple(rows[0][field] for field in ACTION_SCALING_IDENTITY_FIELDS)
    if any(
        tuple(row[field] for field in ACTION_SCALING_IDENTITY_FIELDS) != identity
        for row in rows[1:]
    ):
        raise ValueError(f"{input_path} contains inconsistent metadata")
    action_counts = tuple(map(int, rows[0]["action_counts"].split(",")))
    replicates = int(rows[0]["replicates"])
    first = rows[0]
    spec = AdversarialScalingSpec(
        environment=first["environment"],
        feedback_mode=first["feedback_mode"],
        algorithm_name=first["algorithm"],
        action_counts=action_counts,
        replicates=replicates,
        horizon=int(first["horizon"]),
        environment_seed=int(first["base_environment_seed"] or 0),
        learner_seed=int(first["base_learner_seed"]),
        implementation_version=int(first["implementation_version"]),
        runtime_environment=first["runtime_environment"],
    )
    if spec.runtime_fingerprint != first["runtime_fingerprint"]:
        raise ValueError(f"{input_path} contains an invalid runtime fingerprint")
    if first["run_id"] != spec.run_id:
        raise ValueError(f"{input_path} contains an invalid run identity")
    expected = [
        (n_actions, replicate)
        for n_actions in action_counts
        for replicate in range(replicates)
    ]
    observed = [(int(row["n_actions"]), int(row["replicate"])) for row in rows]
    if observed != expected:
        raise ValueError(f"{input_path} contains incomplete scaling results")
    for row in rows:
        replicate = int(row["replicate"])
        expected_learner_seed = domain_separated_seed(
            int(row["base_learner_seed"]),
            replicate,
            LEARNER_SEED_DOMAIN,
        )
        if int(row["learner_seed"]) != expected_learner_seed:
            raise ValueError(f"{input_path} contains an invalid learner seed schedule")
        if row["environment_seed"]:
            expected_environment_seed = domain_separated_seed(
                int(row["base_environment_seed"]),
                replicate,
                ENVIRONMENT_SEED_DOMAIN,
            )
            if int(row["environment_seed"]) != expected_environment_seed:
                raise ValueError(f"{input_path} contains an invalid environment seed schedule")
        if row["target_regret"] != TARGET_REGRET_BY_ALGORITHM[row["algorithm"]]:
            raise ValueError(f"{input_path} contains an invalid target regret")
        for name in REGRET_FIELDNAMES:
            if not np.isfinite(float(row[name])):
                raise ValueError(f"{input_path} contains non-finite regret")
    return rows
