JOINT_ACTION_HISTOGRAM_FIELD = "joint_action_histograms"

BASE_FIELDNAMES = [
    "run_id",
    "runtime_environment",
    "runtime_fingerprint",
    "feedback_mode",
    "seed",
    "replicate",
    "stationary_method",
    "game",
    "game_payoff_digest",
    "algorithm",
    "algorithm_profile",
    "horizon",
    "t",
    "player",
    "action",
    "payoff",
]

REGRET_NAMES = ("external", "internal", "swap")
REGRET_FIELDNAMES = [
    field for name in REGRET_NAMES
    for field in (f"{name}_regret", f"average_{name}_regret")
]


def regret_fieldnames() -> list[str]:
    return BASE_FIELDNAMES + REGRET_FIELDNAMES + [JOINT_ACTION_HISTOGRAM_FIELD]
