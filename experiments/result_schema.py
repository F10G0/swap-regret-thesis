BASE_FIELDNAMES = [
    "run_id",
    "feedback_mode",
    "regret_evaluation",
    "seed",
    "replicate",
    "stationary_method",
    "game",
    "algorithm",
    "n_players",
    "algorithm_profile",
    "player_algorithm",
    "algorithm_player_0",
    "algorithm_player_1",
    "horizon",
    "t",
    "player",
    "action",
    "payoff",
]

REGRET_NAMES = ("external", "internal", "swap")
REGRET_EVALUATIONS = ("expected", "realized", "both")


def _regret_fieldnames(regret_type: str) -> list[str]:
    return [field for name in REGRET_NAMES for field in (f"{regret_type}_{name}_regret", f"average_{regret_type}_{name}_regret")]


EXPECTED_REGRET_FIELDNAMES = _regret_fieldnames("expected")
REALIZED_REGRET_FIELDNAMES = _regret_fieldnames("realized")


def default_regret_evaluation(feedback_mode: str) -> str:
    if feedback_mode == "full_information":
        return "expected"
    if feedback_mode == "bandit":
        return "realized"
    raise ValueError(f"unknown feedback mode: {feedback_mode}")


def resolve_regret_evaluation(feedback_mode: str, regret_evaluation: str) -> str:
    if regret_evaluation == "feedback_aligned":
        return default_regret_evaluation(feedback_mode)
    if regret_evaluation not in REGRET_EVALUATIONS:
        raise ValueError(f"unknown regret evaluation: {regret_evaluation}")
    return regret_evaluation


def regret_sources(regret_evaluation: str) -> tuple[str, ...]:
    if regret_evaluation == "both":
        return ("expected", "realized")
    if regret_evaluation in REGRET_EVALUATIONS:
        return (regret_evaluation,)
    raise ValueError(f"unknown regret evaluation: {regret_evaluation}")


def regret_fieldnames(regret_evaluation: str) -> list[str]:
    source_fields = {
        "expected": EXPECTED_REGRET_FIELDNAMES,
        "realized": REALIZED_REGRET_FIELDNAMES,
    }
    return BASE_FIELDNAMES + [
        field
        for source in regret_sources(regret_evaluation)
        for field in source_fields[source]
    ]
