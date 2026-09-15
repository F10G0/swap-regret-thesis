import numpy as np


def aggregate_adversarial_regret(
    trajectories: list[list[dict[str, str]]],
    column: str,
    scale_by_sqrt_time: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows_by_time = [{int(row["t"]): row for row in trajectory} for trajectory in trajectories]
    # Use exactly observed, shared timestamps when recording budgets differ.
    times = np.asarray(sorted(set.intersection(*(set(rows) for rows in rows_by_time))), dtype=int)
    values = np.asarray(
        [[float(rows[time][column]) for time in times] for rows in rows_by_time]
    )
    if scale_by_sqrt_time:
        values = values / np.sqrt(times)
    return (
        times,
        np.mean(values, axis=0),
    )
