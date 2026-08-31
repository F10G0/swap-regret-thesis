"""Read immutable capabilities carried by the installed build."""

from importlib.util import find_spec


EXPERIMENTAL_TRAJECTORIES_MARKER = "swap_regret_experimental_trajectories_enabled"


def experimental_trajectories_built() -> bool:
    """Whether this installed distribution contains trajectory support."""
    return find_spec(EXPERIMENTAL_TRAJECTORIES_MARKER) is not None


__all__ = ["experimental_trajectories_built"]
