"""Validation for explicitly requested trajectory renders."""


DEFAULT_FINAL_INTERVAL_SEGMENTS = 10
MIN_FINAL_INTERVAL_SEGMENTS = 1
MAX_FINAL_INTERVAL_SEGMENTS = 50
DEFAULT_TRAJECTORY_COMPARISON_VIEW = "geometry"
TRAJECTORY_COMPARISON_VIEWS = frozenset({"geometry", "unified"})


def parse_final_interval_segments(value: str | None) -> int:
    if value is None:
        return DEFAULT_FINAL_INTERVAL_SEGMENTS
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "final interval segments must be an integer"
        ) from error
    if number < MIN_FINAL_INTERVAL_SEGMENTS:
        raise ValueError("final interval segments must be positive")
    if number > MAX_FINAL_INTERVAL_SEGMENTS:
        raise ValueError(
            "final interval segments must not exceed "
            f"{MAX_FINAL_INTERVAL_SEGMENTS}"
        )
    return number


def parse_focus_final_interval(value: str | None) -> bool:
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError("focus_final_interval must be 0 or 1")


def parse_trajectory_comparison_view(value: str | None) -> str:
    view = (
        DEFAULT_TRAJECTORY_COMPARISON_VIEW
        if value is None
        else value
    )
    if view not in TRAJECTORY_COMPARISON_VIEWS:
        raise ValueError(
            "comparison_view must be 'geometry' or 'unified'"
        )
    return view
