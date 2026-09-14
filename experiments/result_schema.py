RESULT_IMPLEMENTATION_VERSION = 6

BASE_FIELDNAMES = [
    "run_id",
    "implementation_version",
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
    return BASE_FIELDNAMES + REGRET_FIELDNAMES


def result_implementation_version(row: dict[str, str]) -> int:
    version = int(row.get("implementation_version") or 0)
    if version != RESULT_IMPLEMENTATION_VERSION:
        raise ValueError(
            f"incompatible result implementation_version {version}; "
            f"expected {RESULT_IMPLEMENTATION_VERSION}. Re-run the experiment; "
            "existing results are not migrated."
        )
    return version
