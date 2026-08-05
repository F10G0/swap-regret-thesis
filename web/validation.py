from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from experiments.result_schema import REGRET_EVALUATIONS, default_regret_evaluation


@dataclass(frozen=True)
class ExperimentForm:
    game: str
    feedback_mode: str
    algorithm_names: tuple[str, ...]
    horizon: int
    seed: int
    replicates: int
    regret_evaluation: str = "feedback_aligned"


@dataclass(frozen=True)
class AdversarialExperimentForm:
    algorithm_name: str
    n_actions: int
    memory_window: int
    horizon: int
    seed: int


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


def _form_algorithm_names(values: Mapping[str, str]) -> tuple[str, ...]:
    getlist = getattr(values, "getlist", None)
    if getlist is not None:
        names = tuple(getlist("algorithm_names"))
        if names:
            return names
    generic = values.get("algorithm_names")
    if isinstance(generic, (list, tuple)):
        return tuple(generic)
    if generic:
        return (generic,)

    names = []
    player = 0
    while f"algorithm_player_{player}" in values:
        names.append(values[f"algorithm_player_{player}"])
        player += 1
    if not names:
        raise ValueError("missing form field: algorithm_names")
    return tuple(names)


def parse_experiment_form(
    values: Mapping[str, str],
    games: set[str] | Mapping[str, int],
    algorithms_by_feedback_mode: dict[str, list[str]],
    max_horizon: int,
    max_replicates: int = 100,
) -> ExperimentForm:
    try:
        game = values["game"]
        feedback_mode = values["feedback_mode"]
        horizon_value = values["horizon"]
        seed_value = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    if game not in games:
        raise ValueError(f"unknown game: {game}")
    if feedback_mode not in algorithms_by_feedback_mode:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")
    regret_evaluation = values.get("regret_evaluation", default_regret_evaluation(feedback_mode))
    if regret_evaluation not in REGRET_EVALUATIONS:
        raise ValueError(f"unknown regret evaluation: {regret_evaluation}")

    algorithm_names = _form_algorithm_names(values)
    expected_players = games[game] if isinstance(games, Mapping) else 2
    if len(algorithm_names) != expected_players:
        raise ValueError(f"game {game} requires {expected_players} player algorithms")
    available_algorithms = algorithms_by_feedback_mode[feedback_mode]
    for algorithm_name in algorithm_names:
        if algorithm_name not in available_algorithms:
            raise ValueError(f"algorithm {algorithm_name} is not available for {feedback_mode}")

    if feedback_mode == "bandit":
        replicates = parse_positive_integer(
            values.get("replicates", ""),
            "replicates",
            max_replicates,
        )
    else:
        replicates = 1
    return ExperimentForm(
        game=game,
        feedback_mode=feedback_mode,
        algorithm_names=algorithm_names,
        horizon=parse_positive_integer(horizon_value, "horizon", max_horizon),
        seed=parse_non_negative_integer(seed_value, "seed"),
        replicates=replicates,
        regret_evaluation=regret_evaluation,
    )


def parse_adversarial_experiment_form(
    values: Mapping[str, str],
    algorithms: set[str] | list[str] | tuple[str, ...],
    max_actions: int,
    max_horizon: int,
) -> AdversarialExperimentForm:
    try:
        algorithm_name = values["algorithm_name"]
        n_actions = values["n_actions"]
        memory_window = values["memory_window"]
        horizon = values["horizon"]
        seed = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    if algorithm_name not in algorithms:
        raise ValueError(
            f"algorithm {algorithm_name} is not available for the adversarial environment"
        )
    action_count = parse_positive_integer(
        n_actions,
        "number of actions",
        max_actions,
    )
    if action_count < 2:
        raise ValueError("number of actions must be at least 2")
    window = parse_non_negative_integer(memory_window, "memory window")
    if window > max_horizon:
        raise ValueError(f"memory window must not exceed {max_horizon}")
    return AdversarialExperimentForm(
        algorithm_name=algorithm_name,
        n_actions=action_count,
        memory_window=window,
        horizon=parse_positive_integer(horizon, "horizon", max_horizon),
        seed=parse_non_negative_integer(seed, "seed"),
    )
