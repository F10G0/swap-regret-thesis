ALGORITHM_LABELS = {
    "hedge": "Hedge",
    "optimistic_hedge": "OptHedge",
    "auer_exp3": "EXP3",
    "exp3_ix": "EXP3-IX",
    "bm_hedge": "BM-Hedge",
    "bm_optimistic_hedge": "BM-OptHedge",
    "bm_exp3": "BM-EXP3",
    "ito_hedge": "Ito-Hedge",
    "ito_tsallis": "Ito-Tsallis",
    "lce_ix": "LCE-IX",
    "regret_matching": "RM",
    "stationary_regret_matching": "SRM",
}


def algorithm_label(name: str) -> str:
    return ALGORITHM_LABELS.get(name, name.replace("_", " "))


def algorithm_profile_label(names) -> str:
    return " vs ".join(algorithm_label(name) for name in names)
