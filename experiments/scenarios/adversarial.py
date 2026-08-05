import csv
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from config import (
    ADVERSARIAL_ACTIONS,
    ADVERSARIAL_MEMORY_WINDOW,
    ADVERSARIAL_RAW_DIR,
    HORIZON,
    SEED,
)
from environments import HistoricalFrequencyAdversary
from experiments.recorder import CsvRecorder
from experiments.result_schema import (
    EXPECTED_REGRET_FIELDNAMES,
    REALIZED_REGRET_FIELDNAMES,
)
from experiments.runner import ExperimentCancelled
from experiments.scenarios.full_information_cross_play import ALGORITHMS
from metrics.regret import RegretBundles


ENVIRONMENT_ID = "historical_frequency_v2"
MAX_ADVERSARIAL_ACTIONS = 100
TARGET_REGRET_BY_ALGORITHM = {
    "hedge": "external",
    "bm": "swap",
    "ito": "swap",
    "regret_matching": "internal",
    "stationary_regret_matching": "internal",
}
ADVERSARIAL_RESULT_FIELDNAMES = [
    "run_id",
    "environment",
    "n_actions",
    "memory",
    "algorithm",
    "horizon",
    "seed",
    "t",
    "action",
    "punished_action",
    "payoff",
    *EXPECTED_REGRET_FIELDNAMES,
    *REALIZED_REGRET_FIELDNAMES,
]
ADVERSARIAL_IDENTITY_FIELDS = (
    "run_id",
    "environment",
    "n_actions",
    "memory",
    "algorithm",
    "horizon",
    "seed",
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


@dataclass(frozen=True)
class AdversarialExperimentSpec:
    algorithm_name: str
    n_actions: int
    horizon: int
    seed: int
    memory_window: int = ADVERSARIAL_MEMORY_WINDOW

    def __post_init__(self) -> None:
        if not self.algorithm_name:
            raise ValueError("algorithm name must not be empty")
        if not 2 <= self.n_actions <= MAX_ADVERSARIAL_ACTIONS:
            raise ValueError(f"number of actions must be between 2 and {MAX_ADVERSARIAL_ACTIONS}")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.memory_window < 0:
            raise ValueError("memory window must be non-negative")

    def configuration(self) -> dict:
        return {
            "environment": ENVIRONMENT_ID,
            "n_actions": self.n_actions,
            "memory": adversarial_memory_key(self.memory_window),
            "algorithm": self.algorithm_name,
            "horizon": self.horizon,
            "seed": self.seed,
        }

    @property
    def run_id(self) -> str:
        payload = json.dumps(self.configuration(), sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()[:10]
        return f"historical_frequency_{self.n_actions}a_{self.algorithm_name}_{digest}"


def run_adversarial_experiment(
    algorithm_name: str,
    n_actions: int = ADVERSARIAL_ACTIONS,
    horizon: int = HORIZON,
    seed: int = SEED,
    memory_window: int = ADVERSARIAL_MEMORY_WINDOW,
    output_dir: str | Path = ADVERSARIAL_RAW_DIR,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"unknown full-information algorithm: {algorithm_name}")
    spec = AdversarialExperimentSpec(
        algorithm_name=algorithm_name,
        n_actions=n_actions,
        horizon=horizon,
        seed=seed,
        memory_window=memory_window,
    )
    output_path = Path(output_dir) / f"{spec.run_id}.csv"
    if output_path.exists():
        raise FileExistsError(
            f"adversarial experiment {spec.run_id} already exists at {output_path}"
        )

    environment = HistoricalFrequencyAdversary(
        spec.n_actions,
        memory_window=spec.memory_window or None,
    )
    learner = ALGORITHMS[spec.algorithm_name].create(
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
            environment.step((action,))
            payoffs = environment.feedback()
            regrets.update(strategy, action, payoffs)
            learner.update(payoffs)

            recorder.record(
                {
                    **metadata,
                    "t": time,
                    "action": action,
                    "punished_action": environment.punished_action,
                    "payoff": float(payoffs[action]),
                    **regrets.expected.summary(time),
                    **regrets.realized.summary(time),
                }
            )

    return output_path


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
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(ADVERSARIAL_RESULT_FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"{input_path} is missing required columns: {columns}")
        for row in reader:
            adversarial_memory_window(row["memory"])
            identity = tuple(row[field] for field in ADVERSARIAL_IDENTITY_FIELDS)
            if expected_identity is None:
                expected_identity = identity
            elif identity != expected_identity:
                raise ValueError(f"{input_path} contains inconsistent metadata")

            time = int(row["t"])
            horizon = int(row["horizon"])
            n_actions = int(row["n_actions"])
            action = int(row["action"])
            punished_action = int(row["punished_action"])
            if time != expected_time or horizon <= 0 or time > horizon:
                raise ValueError(f"{input_path} contains invalid round metadata")
            if not 0 <= action < n_actions or not 0 <= punished_action < n_actions:
                raise ValueError(f"{input_path} contains an invalid action")

            stride = 1 if max_points is None else max(1, (horizon + max_points - 1) // max_points)
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
    with input_path.open("rb") as file:
        header = file.readline()
        if not header:
            raise ValueError(f"{input_path} is empty")
        data_start = file.tell()
        file.seek(0, 2)
        position = file.tell()
        buffer = b""
        last_line = None
        while position > data_start:
            chunk_start = max(data_start, position - 8_192)
            file.seek(chunk_start)
            buffer = file.read(position - chunk_start) + buffer
            position = chunk_start
            lines = buffer.splitlines()
            if len(lines) >= 2 or position == data_start:
                last_line = lines[-1] if lines else None
                break

    if not last_line:
        raise ValueError(f"{input_path} is empty")
    reader = csv.DictReader([header.decode("utf-8"), last_line.decode("utf-8")])
    missing = set(ADVERSARIAL_RESULT_FIELDNAMES) - set(reader.fieldnames or ())
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{input_path} is missing required columns: {columns}")
    row = next(reader)
    adversarial_memory_window(row["memory"])
    horizon = int(row["horizon"])
    n_actions = int(row["n_actions"])
    if int(row["t"]) != horizon or horizon <= 0:
        raise ValueError(f"{input_path} has no complete final round")
    if not 0 <= int(row["action"]) < n_actions:
        raise ValueError(f"{input_path} contains an invalid action")
    if not 0 <= int(row["punished_action"]) < n_actions:
        raise ValueError(f"{input_path} contains an invalid punished action")
    return row
