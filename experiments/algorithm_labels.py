ALGORITHM_LABELS = {
    "hedge": "Hedge",
    "exp3": "EXP3",
    "exp3_ix": "EXP3-IX",
    "bm": "BM",
    "ito": "Ito",
    "lce_ix": "LCE-IX",
    "regret_matching": "RM",
    "stationary_regret_matching": "SRM",
}


def algorithm_label(name: str) -> str:
    return ALGORITHM_LABELS.get(name, name.replace("_", " "))


def algorithm_profile_label(names) -> str:
    return " vs ".join(algorithm_label(name) for name in names)
