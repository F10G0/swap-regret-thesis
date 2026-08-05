import csv
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import re

from experiments.result_schema import (
    EXPECTED_REGRET_FIELDNAMES,
    REALIZED_REGRET_FIELDNAMES,
    REGRET_NAMES,
    default_regret_evaluation,
    regret_sources,
)
from experiments.recorder import read_final_csv_rows


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
PAYOFF_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def regret_column(regret_type: str, regret_name: str) -> str:
    return f"{regret_type}_{regret_name}_regret"


def average_regret_column(regret_type: str, regret_name: str) -> str:
    return f"average_{regret_column(regret_type, regret_name)}"


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


def result_game_payoff_digest(row: dict[str, str]) -> str:
    digest = row.get("game_payoff_digest", "").strip()
    if not digest or digest == "0":
        return ""
    if not PAYOFF_DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("invalid game_payoff_digest")
    return digest


def _row_identity(row: dict[str, str]) -> tuple:
    return (
        *(row[column] for column in IDENTITY_COLUMNS),
        result_game_payoff_digest(row),
        result_regret_evaluation(row),
        *result_algorithm_profile(row),
    )


def _validated_round(
    input_path: Path,
    time: int,
    rows: list[dict[str, str]],
    n_players: int,
) -> Iterator[dict[str, str]]:
    players = [int(row["player"]) for row in rows]
    if len(players) != n_players or set(players) != set(range(n_players)):
        raise ValueError(
            f"{input_path} round {time} must contain exactly one row for every player"
        )
    yield from rows


def _validated_rows(
    input_path: Path,
    rows: Iterable[dict[str, str]],
    fieldnames: set[str],
    *,
    require_complete_trajectory: bool,
) -> Iterator[dict[str, str]]:
    missing_columns = BASE_RESULT_COLUMNS - fieldnames
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")
    if "algorithm_profile" not in fieldnames and not LEGACY_ALGORITHM_COLUMNS <= fieldnames:
        raise ValueError(f"{input_path} is missing required algorithm profile columns")

    expected_identity = None
    expected_horizon = None
    n_players = None
    first_time = None
    current_time = None
    current_rows: list[dict[str, str]] = []
    for row in rows:
        regret_evaluation = result_regret_evaluation(row)
        row["regret_evaluation"] = regret_evaluation
        identity = _row_identity(row)
        if expected_identity is None:
            missing_columns = required_columns(row["feedback_mode"], regret_evaluation) - fieldnames
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{input_path} is missing required columns: {missing}")
            expected_identity = identity
            expected_horizon = int(row["horizon"])
            n_players = len(result_algorithm_profile(row))
        elif identity != expected_identity:
            raise ValueError(f"{input_path} contains inconsistent run metadata")

        horizon = int(row["horizon"])
        time = int(row["t"])
        player = int(row["player"])
        if horizon <= 0 or time <= 0 or time > horizon or not 0 <= player < n_players:
            raise ValueError(f"{input_path} contains invalid round metadata")

        if current_time is None:
            first_time = current_time = time
        elif time != current_time:
            if time <= current_time:
                raise ValueError(f"{input_path} rounds are not strictly increasing")
            if require_complete_trajectory and time != current_time + 1:
                raise ValueError(
                    f"{input_path} has a gap between rounds {current_time} and {time}"
                )
            yield from _validated_round(
                input_path, current_time, current_rows, n_players
            )
            current_time = time
            current_rows = []
        current_rows.append(row)

    if current_time is None:
        return

    yield from _validated_round(input_path, current_time, current_rows, n_players)
    if require_complete_trajectory:
        if first_time != 1 or current_time != expected_horizon:
            raise ValueError(
                f"{input_path} must contain every round from 1 through {expected_horizon}"
            )
    elif current_time != expected_horizon:
        raise ValueError(f"{input_path} has no complete final-horizon round")


def iter_result_rows(input_path: str | Path) -> Iterator[dict[str, str]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        yield from _validated_rows(
            input_path,
            reader,
            set(reader.fieldnames or []),
            require_complete_trajectory=True,
        )


def load_final_result_rows(input_path: str | Path) -> list[dict[str, str]]:
    """Load every player row from the final round without scanning the complete file."""
    input_path = Path(input_path)
    fieldnames, rows = read_final_csv_rows(input_path, "t")
    return list(_validated_rows(input_path, rows, fieldnames, require_complete_trajectory=False))
