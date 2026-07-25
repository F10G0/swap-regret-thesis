import csv
from collections.abc import Iterator
from pathlib import Path

from experiments.scenarios.fieldnames import REGRET_NAMES


IDENTITY_COLUMNS = (
    "run_id",
    "game",
    "feedback_mode",
    "algorithm",
    "algorithm_player_0",
    "algorithm_player_1",
    "horizon",
    "seed",
    "replicate",
    "stationary_method",
)

BASE_RESULT_COLUMNS = set(IDENTITY_COLUMNS) | {"t", "player"}
EXPERIMENT_PLAYERS = 2


def regret_column(regret_type: str, regret_name: str) -> str:
    return f"{regret_type}_{regret_name}_regret"


def average_regret_column(regret_type: str, regret_name: str) -> str:
    return f"average_{regret_column(regret_type, regret_name)}"


def regret_type(feedback_mode: str) -> str:
    if feedback_mode == "full_information":
        return "expected"
    if feedback_mode == "bandit":
        return "realized"
    raise ValueError(f"unknown feedback mode: {feedback_mode}")


def regret_columns(feedback_mode: str) -> tuple[str, ...]:
    source = regret_type(feedback_mode)
    return tuple(field for name in REGRET_NAMES for field in (regret_column(source, name), average_regret_column(source, name)))


def required_columns(feedback_mode: str) -> set[str]:
    return BASE_RESULT_COLUMNS | set(regret_columns(feedback_mode))


def _validated_rows(input_path: Path, reader: csv.DictReader) -> Iterator[dict[str, str]]:
    fieldnames = set(reader.fieldnames or [])
    missing_columns = BASE_RESULT_COLUMNS - fieldnames
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    expected_identity = None
    for row in reader:
        identity = tuple(row[column] for column in IDENTITY_COLUMNS)
        if expected_identity is None:
            missing_columns = required_columns(row["feedback_mode"]) - fieldnames
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{input_path} is missing required columns: {missing}")
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(f"{input_path} contains inconsistent run metadata")

        horizon = int(row["horizon"])
        time = int(row["t"])
        player = int(row["player"])
        if horizon <= 0 or time <= 0 or time > horizon or player < 0:
            raise ValueError(f"{input_path} contains invalid round metadata")
        yield row


def iter_result_rows(input_path: str | Path) -> Iterator[dict[str, str]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as file:
        yield from _validated_rows(input_path, csv.DictReader(file))


def load_final_result_rows(input_path: str | Path) -> list[dict[str, str]]:
    """Load the final two-player round without scanning the complete result file."""
    input_path = Path(input_path)
    with input_path.open("rb") as file:
        header = file.readline()
        data_start = file.tell()
        file.seek(0, 2)
        position = file.tell()
        buffer = b""

        while position > data_start and buffer.count(b"\n") <= EXPERIMENT_PLAYERS:
            chunk_start = max(data_start, position - 8192)
            file.seek(chunk_start)
            buffer = file.read(position - chunk_start) + buffer
            position = chunk_start

    lines = buffer.splitlines()[-EXPERIMENT_PLAYERS:]
    reader = csv.DictReader([header.decode("utf-8"), *(line.decode("utf-8") for line in lines)])
    return list(_validated_rows(input_path, reader))
