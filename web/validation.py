from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ExperimentForm:
    game: str
    feedback_mode: str
    algorithm_player_0: str
    algorithm_player_1: str
    horizon: int
    seed: int
    replicate: int
    replicates: int

    @property
    def algorithm_names(self) -> list[str]:
        return [self.algorithm_player_0, self.algorithm_player_1]


def parse_positive_integer(
    value: str,
    field_name: str,
    maximum: int | None = None,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer") from error

    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum}")
    return number


def parse_non_negative_integer(value: str, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer") from error

    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def validate_leaf_filename(filename: str, suffix: str) -> str:
    filename_path = Path(filename)
    if (
        not filename
        or filename_path.name != filename
        or filename_path.suffix.lower() != suffix.lower()
    ):
        raise ValueError("invalid filename")
    return filename


def parse_experiment_form(
    values: Mapping[str, str],
    games: set[str],
    algorithms_by_feedback_mode: dict[str, list[str]],
    max_horizon: int,
    max_replicates: int = 100,
) -> ExperimentForm:
    try:
        game = values["game"]
        feedback_mode = values["feedback_mode"]
        algorithm_player_0 = values["algorithm_player_0"]
        algorithm_player_1 = values["algorithm_player_1"]
        horizon_value = values["horizon"]
        seed_value = values["seed"]
        replicate_value = values["replicate"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    if game not in games:
        raise ValueError(f"unknown game: {game}")
    if feedback_mode not in algorithms_by_feedback_mode:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")

    available_algorithms = algorithms_by_feedback_mode[feedback_mode]
    for algorithm_name in [algorithm_player_0, algorithm_player_1]:
        if algorithm_name not in available_algorithms:
            raise ValueError(
                f"algorithm {algorithm_name} is not available for {feedback_mode}"
            )

    replicates = parse_positive_integer(values.get("replicates", "1"), "replicates", max_replicates) if feedback_mode == "bandit" else 1
    return ExperimentForm(
        game=game,
        feedback_mode=feedback_mode,
        algorithm_player_0=algorithm_player_0,
        algorithm_player_1=algorithm_player_1,
        horizon=parse_positive_integer(horizon_value, "horizon", max_horizon),
        seed=parse_non_negative_integer(seed_value, "seed"),
        replicate=parse_non_negative_integer(replicate_value, "replicate"),
        replicates=replicates,
    )
