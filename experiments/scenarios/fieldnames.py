BASE_FIELDNAMES = [
    "run_id",
    "feedback_mode",
    "seed",
    "replicate",
    "stationary_method",
    "game",
    "algorithm",
    "algorithm_player_0",
    "algorithm_player_1",
    "horizon",
    "t",
    "player",
    "action",
    "payoff",
]

REGRET_NAMES = ("external", "internal", "swap")


def _regret_fieldnames(regret_type: str) -> list[str]:
    return [field for name in REGRET_NAMES for field in (f"{regret_type}_{name}_regret", f"average_{regret_type}_{name}_regret")]


EXPECTED_REGRET_FIELDNAMES = _regret_fieldnames("expected")
REALIZED_REGRET_FIELDNAMES = _regret_fieldnames("realized")


def regret_fieldnames(feedback_mode: str) -> list[str]:
    regret_fields = EXPECTED_REGRET_FIELDNAMES if feedback_mode == "full_information" else REALIZED_REGRET_FIELDNAMES
    return BASE_FIELDNAMES + regret_fields
