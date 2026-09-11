import csv
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import re

from experiments.result_schema import (
    REGRET_FIELDNAMES,
    result_implementation_version,
)
from experiments.recorder import read_final_csv_rows, require_csv_columns
from experiments.recording import action_block_length
from experiments.runtime_environment import (
    runtime_environment_fingerprint,
    validate_runtime_environment,
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

CONSTANT_RESULT_COLUMNS = IDENTITY_COLUMNS + (
    "implementation_version",
    "runtime_environment",
    "runtime_fingerprint",
    "game_payoff_digest",
    "algorithm_profile",
    "n_players",
    "algorithm_player_0",
    "algorithm_player_1",
)

BASE_RESULT_COLUMNS = set(IDENTITY_COLUMNS) | {"t", "player"}
LEGACY_ALGORITHM_COLUMNS = {"algorithm_player_0", "algorithm_player_1"}
PAYOFF_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def regret_column(regret_name: str) -> str:
    return f"{regret_name}_regret"


def average_regret_column(regret_name: str) -> str:
    return f"average_{regret_column(regret_name)}"


def required_columns() -> set[str]:
    return BASE_RESULT_COLUMNS | set(REGRET_FIELDNAMES)


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


def result_runtime_environment(row: dict[str, str]) -> str:
    serialized = row.get("runtime_environment", "").strip()
    if not serialized or serialized == "0":
        return ""
    return validate_runtime_environment(serialized)


def result_runtime_fingerprint(row: dict[str, str]) -> str:
    environment = result_runtime_environment(row)
    fingerprint = row.get("runtime_fingerprint", "").strip()
    if not environment:
        if not fingerprint or fingerprint == "0":
            return ""
        raise ValueError("runtime_fingerprint requires runtime_environment")
    if not PAYOFF_DIGEST_PATTERN.fullmatch(fingerprint):
        raise ValueError("invalid runtime_fingerprint")
    if runtime_environment_fingerprint(environment) != fingerprint:
        raise ValueError("runtime_fingerprint does not match runtime_environment")
    return fingerprint


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
    require_csv_columns(input_path, fieldnames, BASE_RESULT_COLUMNS)
    if "algorithm_profile" not in fieldnames and not LEGACY_ALGORITHM_COLUMNS <= fieldnames:
        raise ValueError(f"{input_path} is missing required algorithm profile columns")

    expected_identity = None
    expected_horizon = None
    n_players = None
    first_time = None
    current_time = None
    previous_time = 0
    current_rows: list[dict[str, str]] = []
    for row in rows:
        # Fully validate constants once; compare their original CSV strings on
        # every subsequent row.
        identity = tuple(row.get(column) for column in CONSTANT_RESULT_COLUMNS)
        if expected_identity is None:
            result_implementation_version(row)
            require_csv_columns(input_path, fieldnames, required_columns())
            if row["feedback_mode"] not in {"full_information", "bandit"}:
                raise ValueError(f"unknown feedback mode: {row['feedback_mode']}")
            result_runtime_fingerprint(row)
            result_game_payoff_digest(row)
            expected_identity = identity
            expected_horizon = int(row["horizon"])
            n_players = len(result_algorithm_profile(row))
            if expected_horizon <= 0 or int(row["seed"]) < 0 or int(row["replicate"]) < 0:
                raise ValueError(f"{input_path} contains invalid run metadata")
        elif identity != expected_identity:
            raise ValueError(f"{input_path} contains inconsistent run metadata")

        time = int(row["t"])
        player = int(row["player"])
        if time <= 0 or time > expected_horizon or not 0 <= player < n_players:
            raise ValueError(f"{input_path} contains invalid round metadata")

        if current_time is None:
            first_time = current_time = time
        elif time != current_time:
            if time <= current_time:
                raise ValueError(f"{input_path} rounds are not strictly increasing")
            yield from _validated_round(
                input_path, current_time, current_rows, n_players
            )
            previous_time, current_time = current_time, time
            current_rows = []
        if "action_history" in fieldnames:
            count = action_block_length(row["action_history"])
            if (require_complete_trajectory and count != time - previous_time) or count > time:
                raise ValueError(f"{input_path} action_history does not cover the checkpoint interval")
        current_rows.append(row)

    if current_time is None:
        return

    yield from _validated_round(input_path, current_time, current_rows, n_players)
    if require_complete_trajectory:
        if first_time != 1 or current_time != expected_horizon:
            raise ValueError(
                f"{input_path} must include checkpoints 1 and {expected_horizon}"
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
