from pathlib import Path


PRECOMPUTED_EQUILIBRIUM_DIR = Path(__file__).resolve().parent / "static" / "equilibria"


def equilibrium_figure_filename(game_name: str, equilibrium: str) -> str:
    return f"{game_name}_{equilibrium}_blue_lower_origin_maximum_profile_weight.png"
