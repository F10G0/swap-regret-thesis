import csv
from collections.abc import Iterator
import json
from pathlib import Path

from experiments.result_schema import (
    EXPECTED_REGRET_FIELDNAMES,
    REALIZED_REGRET_FIELDNAMES,
    REGRET_NAMES,
    default_regret_evaluation,
    regret_sources,
)


IDENTITY_COLUMNS = (
    "run_id",
    "game",
    "feedback_mode",
    "algorithm",
    "horizon",
    "seed",
    "replicate",
    "stationary_method",
)

BASE_RESULT_COLUMNS = set(IDENTITY_COLUMNS) | {"t", "player"}
LEGACY_ALGORITHM_COLUMNS = {"algorithm_player_0", "algorithm_player_1"}


def regret_column(regret_type: str, regret_name: str) -> str:
    return f"{regret_type}_{regret_name}_regret"


def average_regret_column(regret_type: str, regret_name: str) -> str:
    return f"average_{regret_column(regret_type, regret_name)}"


def regret_type(feedback_mode: str) -> str:
    return default_regret_evaluation(feedback_mode)


def regret_columns(regret_evaluation: str) -> tuple[str, ...]:
    return tuple(
        field
        for source in regret_sources(regret_evaluation)
        for name in REGRET_NAMES
        for field in (regret_column(source, name), average_regret_column(source, name))
    )


def required_columns(feedback_mode: str, regret_evaluation: str | None = None) -> set[str]:
    evaluation = default_regret_evaluation(feedback_mode) if regret_evaluation is None else regret_evaluation
    return BASE_RESULT_COLUMNS | set(regret_columns(evaluation))


def result_regret_evaluation(row: dict[str, str]) -> str:
    default_regret_evaluation(row["feedback_mode"])
    present_sources = {
        source
        for source, fields in (
            ("expected", EXPECTED_REGRET_FIELDNAMES),
            ("realized", REALIZED_REGRET_FIELDNAMES),
        )
        if any(field in row for field in fields)
    }
    declared = row.get("regret_evaluation", "").strip()
    if declared:
        declared_sources = set(regret_sources(declared))
        if present_sources != declared_sources:
            raise ValueError("regret_evaluation does not match regret columns")
        return declared
    if present_sources == {"expected", "realized"}:
        return "both"
    if len(present_sources) == 1:
        return present_sources.pop()
    raise ValueError("result has no regret evaluation columns")


def result_algorithm_profile(row: dict[str, str]) -> tuple[str, ...]:
    serialized = row.get("algorithm_profile", "").strip()
    if serialized and serialized != "0":
        try:
            values = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("invalid algorithm_profile JSON") from error
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("algorithm_profile must be a JSON array of algorithm names")
        profile = tuple(values)
        if len(profile) < 2:
            raise ValueError("algorithm_profile must contain at least two algorithms")
        n_players = row.get("n_players", "").strip()
        if n_players and n_players != "0" and int(n_players) != len(profile):
            raise ValueError("n_players does not match algorithm_profile")
        return profile

    if LEGACY_ALGORITHM_COLUMNS <= row.keys():
        return (row["algorithm_player_0"], row["algorithm_player_1"])
    raise ValueError("result has no algorithm profile")


def result_player_algorithm(row: dict[str, str]) -> str:
    player_algorithm = row.get("player_algorithm", "").strip()
    if player_algorithm and player_algorithm != "0":
        return player_algorithm
    profile = result_algorithm_profile(row)
    player = int(row["player"])
    return profile[player] if player < len(profile) else row["algorithm"]


def _row_identity(row: dict[str, str]) -> tuple:
    return tuple(row[column] for column in IDENTITY_COLUMNS) + (result_regret_evaluation(row), *result_algorithm_profile(row))


def _validated_rows(input_path: Path, reader: csv.DictReader) -> Iterator[dict[str, str]]:
    fieldnames = set(reader.fieldnames or [])
    missing_columns = BASE_RESULT_COLUMNS - fieldnames
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")
    if "algorithm_profile" not in fieldnames and not LEGACY_ALGORITHM_COLUMNS <= fieldnames:
        raise ValueError(f"{input_path} is missing required algorithm profile columns")

    expected_identity = None
    for row in reader:
        regret_evaluation = result_regret_evaluation(row)
        row["regret_evaluation"] = regret_evaluation
        identity = _row_identity(row)
        if expected_identity is None:
            missing_columns = required_columns(row["feedback_mode"], regret_evaluation) - fieldnames
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
    """Load every player row from the final round without scanning the complete file."""
    input_path = Path(input_path)
    with input_path.open("rb") as file:
        header = file.readline()
        fieldnames = set(next(csv.reader([header.decode("utf-8")]), []))
        missing_columns = BASE_RESULT_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_path} is missing required columns: {missing}")
        data_start = file.tell()
        file.seek(0, 2)
        position = file.tell()
        buffer = b""

        while True:
            chunk_start = max(data_start, position - 8192)
            file.seek(chunk_start)
            buffer = file.read(position - chunk_start) + buffer
            position = chunk_start
            lines = buffer.splitlines()
            if position > data_start and lines:
                lines = lines[1:]
            if lines:
                rows = list(csv.DictReader([header.decode("utf-8"), *(line.decode("utf-8") for line in lines)]))
                final_time = rows[-1]["t"]
                first_final = len(rows) - 1
                while first_final > 0 and rows[first_final - 1]["t"] == final_time:
                    first_final -= 1
                if first_final > 0 or position == data_start:
                    final_lines = lines[first_final:]
                    reader = csv.DictReader([header.decode("utf-8"), *(line.decode("utf-8") for line in final_lines)])
                    return list(_validated_rows(input_path, reader))
            if position == data_start:
                return []
