from dataclasses import dataclass
from pathlib import Path
import re
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
    environment: str
    initialization_mode: str
    feedback_mode: str
    algorithm_name: str
    n_actions: int
    horizon: int
    environment_seed: int
    learner_seed: int
    replicates: int
    regret_evaluation: str


@dataclass(frozen=True)
class AdversarialScalingForm:
    environment: str
    initialization_mode: str
    feedback_mode: str
    algorithm_name: str
    action_counts: tuple[int, ...]
    replicates: int
    horizon: int
    environment_seed: int
    learner_seed: int
    regret_evaluation: str


def _parse_integer(value: str, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer") from error


def parse_positive_integer(value: str, field_name: str, maximum: int | None = None) -> int:
    number = _parse_integer(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum}")
    return number


def parse_non_negative_integer(value: str, field_name: str) -> int:
    number = _parse_integer(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def parse_action_counts(
    value: str,
    max_actions: int,
    max_values: int = 20,
) -> tuple[int, ...]:
    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if len(tokens) < 2:
        raise ValueError("provide at least two action counts")
    if len(tokens) > max_values:
        raise ValueError(f"provide at most {max_values} action counts")
    action_counts = tuple(
        parse_positive_integer(token, "action count", max_actions)
        for token in tokens
    )
    if any(action_count < 2 for action_count in action_counts):
        raise ValueError("action counts must be at least 2")
    if len(set(action_counts)) != len(action_counts):
        raise ValueError("action counts must be unique")
    return tuple(sorted(action_counts))


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
    else:
        value = values.get("algorithm_names")
        if isinstance(value, (list, tuple)):
            names = tuple(value)
        else:
            names = (value,) if value else ()
    if not names:
        raise ValueError("missing form field: algorithm_names")
    return names


def _validate_learning_configuration(
    feedback_mode: str,
    algorithm_names: tuple[str, ...],
    regret_evaluation: str,
    algorithms_by_feedback_mode: Mapping[str, list[str]],
) -> None:
    if feedback_mode not in algorithms_by_feedback_mode:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")
    if regret_evaluation not in REGRET_EVALUATIONS:
        raise ValueError(f"unknown regret evaluation: {regret_evaluation}")
    for algorithm_name in algorithm_names:
        if algorithm_name not in algorithms_by_feedback_mode[feedback_mode]:
            raise ValueError(f"algorithm {algorithm_name} is not available for {feedback_mode}")


def _parse_learning_configuration(
    values: Mapping[str, str],
    algorithms_by_feedback_mode: Mapping[str, list[str]],
    default_evaluation: str | None = None,
) -> tuple[str, tuple[str, ...], str]:
    try:
        feedback_mode = values["feedback_mode"]
    except KeyError as error:
        raise ValueError("missing form field: feedback_mode") from error
    algorithm_names = _form_algorithm_names(values)
    regret_evaluation = values.get(
        "regret_evaluation",
        default_evaluation or default_regret_evaluation(feedback_mode),
    )
    _validate_learning_configuration(
        feedback_mode,
        algorithm_names,
        regret_evaluation,
        algorithms_by_feedback_mode,
    )
    return feedback_mode, algorithm_names, regret_evaluation


def parse_experiment_form(
    values: Mapping[str, str],
    games: set[str] | Mapping[str, int],
    algorithms_by_feedback_mode: dict[str, list[str]],
    max_horizon: int,
    max_replicates: int = 100,
) -> ExperimentForm:
    try:
        game = values["game"]
        horizon_value = values["horizon"]
        seed_value = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    if game not in games:
        raise ValueError(f"unknown game: {game}")
    feedback_mode, algorithm_names, regret_evaluation = _parse_learning_configuration(
        values,
        algorithms_by_feedback_mode,
    )
    expected_players = games[game] if isinstance(games, Mapping) else 2
    if len(algorithm_names) != expected_players:
        raise ValueError(f"game {game} requires {expected_players} player algorithms")

    replicates = parse_positive_integer(
        values.get("replicates", ""),
        "replicates",
        max_replicates,
    )
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
    algorithms_by_feedback_mode: Mapping[str, list[str]],
    environments: set[str],
    initialization_modes: set[str],
    max_actions: int,
    max_horizon: int,
    max_replicates: int = 100,
) -> AdversarialExperimentForm:
    try:
        environment = values["environment"]
        n_actions = values["n_actions"]
        horizon = values["horizon"]
        learner_seed = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    feedback_mode, algorithm_names, regret_evaluation = _parse_learning_configuration(
        values,
        algorithms_by_feedback_mode,
        default_evaluation="both",
    )
    if len(algorithm_names) != 1:
        raise ValueError("one-player environments require one algorithm")
    algorithm_name = algorithm_names[0]

    initialization_mode = values.get("initialization_mode", "centered")
    environment_seed = values.get("environment_seed", "0")
    if environment not in environments:
        raise ValueError(f"unknown adversarial environment: {environment}")
    if initialization_mode not in initialization_modes:
        raise ValueError(f"unknown initialization mode: {initialization_mode}")
    action_count = parse_positive_integer(
        n_actions,
        "number of actions",
        max_actions,
    )
    if action_count < 2:
        raise ValueError("number of actions must be at least 2")
    return AdversarialExperimentForm(
        environment=environment,
        initialization_mode=initialization_mode,
        feedback_mode=feedback_mode,
        algorithm_name=algorithm_name,
        n_actions=action_count,
        horizon=parse_positive_integer(horizon, "horizon", max_horizon),
        environment_seed=parse_non_negative_integer(
            environment_seed,
            "environment seed",
        ),
        learner_seed=parse_non_negative_integer(learner_seed, "learner seed"),
        replicates=parse_positive_integer(
            values.get("replicates", ""),
            "replicates",
            max_replicates,
        ),
        regret_evaluation=regret_evaluation,
    )


def parse_adversarial_scaling_form(
    values: Mapping[str, str],
    algorithms_by_feedback_mode: Mapping[str, list[str]],
    environments: set[str],
    initialization_modes: set[str],
    max_actions: int,
    max_horizon: int,
    max_replicates: int,
) -> AdversarialScalingForm:
    action_counts = parse_action_counts(
        values.get("scaling_action_counts", ""),
        max_actions,
    )
    common_values = dict(values)
    common_values["n_actions"] = str(action_counts[0])
    common_values["replicates"] = "1"
    common = parse_adversarial_experiment_form(
        common_values,
        algorithms_by_feedback_mode,
        environments,
        initialization_modes,
        max_actions,
        max_horizon,
        max_replicates,
    )
    return AdversarialScalingForm(
        environment=common.environment,
        initialization_mode=common.initialization_mode,
        feedback_mode=common.feedback_mode,
        algorithm_name=common.algorithm_name,
        action_counts=action_counts,
        replicates=parse_positive_integer(
            values.get("scaling_replicates", ""),
            "scaling replicates",
            max_replicates,
        ),
        horizon=common.horizon,
        environment_seed=common.environment_seed,
        learner_seed=common.learner_seed,
        regret_evaluation=common.regret_evaluation,
    )
