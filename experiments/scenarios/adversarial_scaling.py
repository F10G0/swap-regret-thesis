import csv
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np

from experiments.recorder import CsvRecorder, require_csv_columns
from experiments.result_schema import RESULT_IMPLEMENTATION_VERSION, regret_sources, resolve_regret_evaluation
from experiments.runner import ExperimentCancelled
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    INITIALIZATION_LABELS,
    RANDOM_WALK_ENVIRONMENT,
    TARGET_REGRET_BY_ALGORITHM,
    adversarial_environment_detail,
    load_final_adversarial_row,
    run_adversarial_experiment,
)


ACTION_SCALING_IDENTITY_FIELDS = (
    "run_id",
    "implementation_version",
    "environment",
    "initialization_mode",
    "feedback_mode",
    "regret_evaluation",
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
    "expected_regret",
    "realized_regret",
]


@dataclass(frozen=True)
class AdversarialScalingSpec:
    environment: str
    initialization_mode: str
    feedback_mode: str
    algorithm_name: str
    action_counts: tuple[int, ...]
    replicates: int
    horizon: int
    environment_seed: int
    learner_seed: int
    regret_evaluation: str = "both"
    implementation_version: int = RESULT_IMPLEMENTATION_VERSION

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
        object.__setattr__(
            self,
            "regret_evaluation",
            resolve_regret_evaluation(self.feedback_mode, self.regret_evaluation),
        )
        object.__setattr__(self, "action_counts", action_counts)
        for n_actions in action_counts:
            AdversarialExperimentSpec(
                environment=self.environment,
                initialization_mode=self.initialization_mode,
                environment_seed=self.environment_seed,
                feedback_mode=self.feedback_mode,
                algorithm_name=self.algorithm_name,
                n_actions=n_actions,
                horizon=self.horizon,
                seed=self.learner_seed,
                regret_evaluation=self.regret_evaluation,
                implementation_version=self.implementation_version,
            )

    def configuration(self) -> dict:
        random_walk = self.environment == RANDOM_WALK_ENVIRONMENT
        configuration = {
            "environment": self.environment,
            "initialization_mode": self.initialization_mode if random_walk else "",
            "feedback_mode": self.feedback_mode,
            "regret_evaluation": self.regret_evaluation,
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
            "base_environment_seed": self.environment_seed if random_walk else "",
            "base_learner_seed": self.learner_seed,
            "action_counts": ",".join(map(str, self.action_counts)),
            "replicates": self.replicates,
        }
        if self.implementation_version:
            configuration["implementation_version"] = self.implementation_version
        return configuration

    @property
    def run_id(self) -> str:
        identity = self.configuration()
        if self.regret_evaluation == "both":
            identity.pop("regret_evaluation")
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return f"action_scaling_{self.algorithm_name}_{sha256(payload.encode()).hexdigest()[:10]}"


def adversarial_scaling_environment_detail(row: dict[str, str]) -> str:
    if row["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT:
        return adversarial_environment_detail(row)
    return (
        f"{INITIALIZATION_LABELS[row['initialization_mode']]} · "
        f"base environment seed {row['base_environment_seed']}"
    )


def run_adversarial_scaling_experiment(
    spec: AdversarialScalingSpec,
    output_dir: str | Path,
    should_cancel: Callable[[], bool] | None = None,
    completed: Callable[[], None] | None = None,
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
        with CsvRecorder(ACTION_SCALING_FIELDNAMES, output_path) as recorder:
            for n_actions in spec.action_counts:
                for replicate in range(spec.replicates):
                    if should_cancel is not None and should_cancel():
                        raise ExperimentCancelled("experiment cancelled")
                    learner_seed = spec.learner_seed + replicate
                    environment_seed = spec.environment_seed + replicate
                    result_path = run_adversarial_experiment(
                        environment=spec.environment,
                        initialization_mode=spec.initialization_mode,
                        environment_seed=spec.environment_seed,
                        feedback_mode=spec.feedback_mode,
                        algorithm_name=spec.algorithm_name,
                        n_actions=n_actions,
                        horizon=spec.horizon,
                        seed=spec.learner_seed,
                        replicate=replicate,
                        regret_evaluation=spec.regret_evaluation,
                        implementation_version=spec.implementation_version,
                        output_dir=temporary_directory,
                        should_cancel=should_cancel,
                    )
                    final = load_final_adversarial_row(result_path)
                    recorder.record(
                        {
                            **metadata,
                            "n_actions": n_actions,
                            "replicate": replicate,
                            "environment_seed": (
                                environment_seed if final["environment_seed"] else ""
                            ),
                            "learner_seed": learner_seed,
                            "target_regret": target_regret,
                            "expected_regret": final.get(
                                f"expected_{target_regret}_regret",
                                "",
                            ),
                            "realized_regret": final.get(
                                f"realized_{target_regret}_regret",
                                "",
                            ),
                        }
                    )
                    if completed is not None:
                        completed()
    return output_path


def load_adversarial_scaling_rows(input_path: str | Path) -> list[dict[str, str]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        optional_legacy = {"regret_evaluation", "implementation_version"}
        require_csv_columns(input_path, reader.fieldnames or (), set(ACTION_SCALING_FIELDNAMES) - optional_legacy)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    for row in rows:
        row.setdefault("regret_evaluation", "both")
        row.setdefault("implementation_version", "0")

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
        initialization_mode=first["initialization_mode"],
        feedback_mode=first["feedback_mode"],
        algorithm_name=first["algorithm"],
        action_counts=action_counts,
        replicates=replicates,
        horizon=int(first["horizon"]),
        environment_seed=int(first["base_environment_seed"] or 0),
        learner_seed=int(first["base_learner_seed"]),
        regret_evaluation=first["regret_evaluation"],
        implementation_version=int(first["implementation_version"]),
    )
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
        if int(row["learner_seed"]) != int(row["base_learner_seed"]) + replicate:
            raise ValueError(f"{input_path} contains an invalid learner seed schedule")
        if row["environment_seed"] and int(row["environment_seed"]) != int(
            row["base_environment_seed"]
        ) + replicate:
            raise ValueError(f"{input_path} contains an invalid environment seed schedule")
        if row["target_regret"] != TARGET_REGRET_BY_ALGORITHM[row["algorithm"]]:
            raise ValueError(f"{input_path} contains an invalid target regret")
        selected_sources = set(regret_sources(row["regret_evaluation"]))
        for source in selected_sources:
            field = f"{source}_regret"
            if not np.isfinite(float(row[field])):
                raise ValueError(f"{input_path} contains non-finite regret")
        for source in {"expected", "realized"} - selected_sources:
            if row[f"{source}_regret"]:
                raise ValueError(
                    f"{input_path} contains regret outside its selected evaluation"
                )
    return rows
