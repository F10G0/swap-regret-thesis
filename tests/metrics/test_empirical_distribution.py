import numpy as np
import pytest

from metrics.empirical_distribution import (
    default_checkpoints,
    empirical_distribution_trajectory,
    final_interval_checkpoints,
    final_logarithmic_interval_start,
    mean_empirical_distribution_trajectory,
    uniform_checkpoints,
)


def test_cumulative_empirical_distribution_at_manual_checkpoints() -> None:
    actions = [(0, 0), (1, 0), (1, 1), (1, 0)]
    trajectory = empirical_distribution_trajectory(actions, (2, 2), checkpoints=[1, 2, 4])

    expected = np.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.5, 0.0], [0.5, 0.0]],
        [[0.25, 0.0], [0.5, 0.25]],
    ])
    assert np.array_equal(trajectory.horizons, [1, 2, 4])
    assert np.allclose(trajectory.distributions, expected)


def test_three_player_heterogeneous_trajectory_keeps_every_player() -> None:
    actions = [(0, 0, 1), (1, 2, 0), (0, 0, 1)]
    trajectory = empirical_distribution_trajectory(actions, (2, 3, 2), checkpoints=[2, 3])

    assert trajectory.distributions.shape == (2, 2, 3, 2)
    assert trajectory.distributions[0, 0, 0, 1] == pytest.approx(0.5)
    assert trajectory.distributions[0, 1, 2, 0] == pytest.approx(0.5)
    assert trajectory.distributions[1, 0, 0, 1] == pytest.approx(2.0 / 3.0)
    assert np.allclose(np.sum(trajectory.distributions, axis=(1, 2, 3)), 1.0)


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (1, [1]),
        (99, [1, 99]),
        (100, [1, 100]),
        (250, [1, 100, 250]),
        (1_000, [1, 100, 1_000]),
        (12_345, [1, 100, 1_000, 10_000, 12_345]),
    ],
)
def test_default_checkpoints_include_first_round_powers_of_ten_and_final(horizon: int, expected: list[int]) -> None:
    assert np.array_equal(default_checkpoints(horizon), expected)


@pytest.mark.parametrize(
    ("horizon", "count", "expected"),
    [
        (100, 10, [1, 12, 23, 34, 45, 56, 67, 78, 89, 100]),
        (11, 4, [1, 4, 8, 11]),
        (3, 10, [1, 2, 3]),
    ],
)
def test_uniform_checkpoints_span_first_to_final_round_evenly(horizon: int, count: int, expected: list[int]) -> None:
    assert np.array_equal(uniform_checkpoints(horizon, count), expected)


@pytest.mark.parametrize(
    ("horizon", "segments", "expected"),
    [
        (1, 4, [1]),
        (7, 4, [1, 3, 4, 6, 7]),
        (10, 4, [1, 3, 6, 8, 10]),
        (6_500, 4, [1, 10, 100, 1_000, 2_375, 3_750, 5_125, 6_500]),
        (10_000, 4, [1, 10, 100, 1_000, 3_250, 5_500, 7_750, 10_000]),
        (10_000, 1, [1, 10, 100, 1_000, 10_000]),
        (2, 50, [1, 2]),
    ],
)
def test_final_interval_checkpoints_preserve_logs_and_subdivide_final_interval(
    horizon: int,
    segments: int,
    expected: list[int],
) -> None:
    checkpoints = final_interval_checkpoints(horizon, segments)

    assert np.array_equal(checkpoints, expected)
    assert np.array_equal(
        checkpoints,
        final_interval_checkpoints(horizon, segments),
    )
    assert np.all(np.diff(checkpoints) > 0)
    assert checkpoints[-1] == horizon


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [(1, 1), (7, 1), (10, 1), (1_000, 100), (6_500, 1_000)],
)
def test_final_logarithmic_interval_start_is_strictly_before_horizon(
    horizon: int,
    expected: int,
) -> None:
    assert final_logarithmic_interval_start(horizon) == expected


@pytest.mark.parametrize("segments", [0, -1, 1.5])
def test_final_interval_segments_must_be_positive_integers(
    segments,
) -> None:
    with pytest.raises(ValueError, match="final_interval_segments"):
        final_interval_checkpoints(100, segments)


def test_mean_empirical_trajectory_averages_matching_replicates() -> None:
    first = empirical_distribution_trajectory([(0, 0), (0, 0)], (2, 2), [1, 2])
    second = empirical_distribution_trajectory([(1, 1), (1, 1)], (2, 2), [1, 2])

    mean = mean_empirical_distribution_trajectory([first, second])

    assert np.array_equal(mean.horizons, np.array([1, 2]))
    assert np.allclose(mean.distributions[:, 0, 0], 0.5)
    assert np.allclose(mean.distributions[:, 1, 1], 0.5)
