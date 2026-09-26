from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from experiments.result_schema import REGRET_NAMES


@dataclass(frozen=True)
class ProfileSelection:
    mode: str
    context_id: str
    comparison_mode: str
    metric: str
    view: str
    profiles: tuple[str, ...]
    action: str
    horizon: str
    feedback: str


@dataclass(frozen=True)
class FigureSelection:
    mode: str
    context_id: str
    comparison_mode: str
    metric: str
    view: str
    profiles: tuple[str, ...]
    action: str
    horizon: str


def parse_profile_selection(values: Mapping[str, str]) -> ProfileSelection:
    mode = values.get("mode", "")
    context_id = values.get("context_id", "")
    comparison_mode = values.get("comparison_mode", "profiles")
    view = values.get("view", "horizon_scaling" if comparison_mode == "horizons" else "all")
    metric = values.get("metric", "all")
    feedback = values.get("feedback", "both")
    if mode not in {"fixed", "adversarial"}:
        raise ValueError("Unknown experiment mode")
    if not re.fullmatch(r"[0-9a-f]{24}", context_id):
        raise ValueError("Choose an available result set")
    if comparison_mode not in {"profiles", "regrets", "actions", "horizons"}:
        raise ValueError("Unknown comparison mode")
    if feedback not in {"full_information", "bandit", "both"}:
        raise ValueError("Unknown feedback mode")
    if comparison_mode == "actions" and mode != "adversarial":
        raise ValueError("Action-space comparison is available only for one-player results")
    if metric not in {*REGRET_NAMES, "all"} or view not in {"average", "sqrt_scaling", "horizon_scaling", "all"}:
        raise ValueError("Unknown regret metric or view")
    if comparison_mode == "horizons" and view != "horizon_scaling":
        raise ValueError("Horizon comparison requires the horizon-scaling view")
    if comparison_mode != "horizons" and view == "horizon_scaling":
        raise ValueError("The horizon-scaling view is available only for horizon comparison")
    if hasattr(values, "getlist"):
        profiles = values.getlist("profiles")
    else:
        profiles = values.get("profiles", [])
        profiles = [profiles] if isinstance(profiles, str) else profiles
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("Select at least one algorithm profile")
    if len(profiles) > 256 or any(not isinstance(profile, str) or not re.fullmatch(r"[a-z0-9_]+", profile) for profile in profiles):
        raise ValueError("Invalid algorithm profile selection")
    profiles = tuple(sorted(set(profiles)))
    if "all" in profiles and (profiles != ("all",) or comparison_mode not in {"regrets", "horizons"}):
        raise ValueError("All algorithm profiles is available only for regret or horizon comparison")
    if comparison_mode in {"regrets", "actions", "horizons"} and len(profiles) != 1:
        raise ValueError("Select exactly one algorithm profile for this comparison")
    if comparison_mode == "regrets" and metric != "all":
        raise ValueError("Regret-notion comparison includes all regret notions")
    if comparison_mode == "horizons" and metric != "all":
        raise ValueError("Horizon comparison includes all regret notions")
    action = "all" if comparison_mode == "actions" else values.get("action", "")
    if mode == "adversarial" and comparison_mode != "actions" and not re.fullmatch(r"[1-9][0-9]*", action):
        raise ValueError("Choose an available action count")
    horizon = "all" if comparison_mode == "horizons" else values.get("horizon", "")
    if comparison_mode != "horizons" and not re.fullmatch(r"[1-9][0-9]*", horizon):
        raise ValueError("Choose an available horizon")
    return ProfileSelection(mode, context_id, comparison_mode, metric, view, profiles, action, horizon, feedback)


@dataclass(frozen=True)
class ExperimentForm:
    game: str
    feedback_mode: str
    algorithm_names: tuple[str, ...]
    horizon: int
    seed: int
    replicates: int
    horizons: tuple[int, ...] = ()

    @property
    def horizon_values(self) -> tuple[int, ...]:
        return self.horizons or (self.horizon,)


@dataclass(frozen=True)
class AdversarialExperimentForm:
    environment: str
    feedback_mode: str
    algorithm_name: str
    action_counts: tuple[int, ...]
    horizon: int
    seed: int
    replicates: int
    horizons: tuple[int, ...] = ()

    @property
    def horizon_values(self) -> tuple[int, ...]:
        return self.horizons or (self.horizon,)


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


def _integer_list_tokens(value: str, name: str, max_values: int | None = None) -> list[str]:
    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if not tokens:
        raise ValueError(f"provide at least one {name}")
    if max_values is not None and len(tokens) > max_values:
        raise ValueError(f"provide at most {max_values} {name}s")
    return tokens


def parse_action_counts(value: str, max_actions: int, max_values: int = 20) -> tuple[int, ...]:
    tokens = _integer_list_tokens(value, "action count", max_values)
    action_counts = tuple(parse_positive_integer(token, "action count", max_actions) for token in tokens)
    if any(action_count < 2 for action_count in action_counts):
        raise ValueError("action counts must be at least 2")
    if len(set(action_counts)) != len(action_counts):
        raise ValueError("action counts must be unique")
    return action_counts


def parse_horizons(value: str, max_horizon: int, max_values: int | None = None) -> tuple[int, ...]:
    tokens = _integer_list_tokens(value, "horizon", max_values)
    return tuple(sorted({parse_positive_integer(token, "horizon", max_horizon) for token in tokens}))


def validate_leaf_filename(filename: str, suffix: str) -> str:
    filename_path = Path(filename)
    if not filename or filename_path.name != filename or filename_path.suffix.lower() != suffix.lower():
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


def _parse_learning_configuration(values: Mapping[str, str], algorithms_by_feedback_mode: Mapping[str, list[str]]) -> tuple[str, tuple[str, ...]]:
    try:
        feedback_mode = values["feedback_mode"]
    except KeyError as error:
        raise ValueError("missing form field: feedback_mode") from error
    algorithm_names = _form_algorithm_names(values)
    if feedback_mode not in algorithms_by_feedback_mode:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")
    for algorithm_name in algorithm_names:
        if algorithm_name not in algorithms_by_feedback_mode[feedback_mode]:
            raise ValueError(f"algorithm {algorithm_name} is not available for {feedback_mode}")
    return feedback_mode, algorithm_names


def parse_experiment_form(values: Mapping[str, str], games: Mapping[str, int], algorithms_by_feedback_mode: dict[str, list[str]], max_horizon: int, max_replicates: int = 100) -> ExperimentForm:
    try:
        game = values["game"]
        horizon_value = values["horizon"]
        seed_value = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    if game not in games:
        raise ValueError(f"unknown game: {game}")
    feedback_mode, algorithm_names = _parse_learning_configuration(values, algorithms_by_feedback_mode)
    expected_players = games[game]
    if len(algorithm_names) != expected_players:
        raise ValueError(f"game {game} requires {expected_players} player algorithms")

    replicates = parse_positive_integer(values.get("replicates", ""), "replicates", max_replicates)
    horizons = parse_horizons(horizon_value, max_horizon)
    return ExperimentForm(
        game=game,
        feedback_mode=feedback_mode,
        algorithm_names=algorithm_names,
        horizon=horizons[0],
        seed=parse_non_negative_integer(seed_value, "seed"),
        replicates=replicates,
        horizons=horizons,
    )


def parse_adversarial_experiment_form(values: Mapping[str, str], algorithms_by_feedback_mode: Mapping[str, list[str]], environments: set[str], max_actions: int, max_horizon: int, max_replicates: int = 100) -> AdversarialExperimentForm:
    try:
        environment = values["environment"]
        actions = values["actions"]
        horizon = values["horizon"]
        seed = values["seed"]
    except KeyError as error:
        raise ValueError(f"missing form field: {error.args[0]}") from error

    feedback_mode, algorithm_names = _parse_learning_configuration(values, algorithms_by_feedback_mode)
    if len(algorithm_names) != 1:
        raise ValueError("one-player environments require one algorithm")
    algorithm_name = algorithm_names[0]

    if environment not in environments:
        raise ValueError(f"unknown adversarial environment: {environment}")
    action_counts = parse_action_counts(actions, max_actions)
    horizons = parse_horizons(horizon, max_horizon)
    return AdversarialExperimentForm(
        environment=environment,
        feedback_mode=feedback_mode,
        algorithm_name=algorithm_name,
        action_counts=action_counts,
        horizon=horizons[0],
        seed=parse_non_negative_integer(seed, "seed"),
        replicates=parse_positive_integer(values.get("replicates", ""), "replicates", max_replicates),
        horizons=horizons,
    )
