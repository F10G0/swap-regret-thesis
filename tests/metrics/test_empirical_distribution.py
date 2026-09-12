import numpy as np
import pytest

from metrics.empirical_distribution import (
    default_checkpoints,
    empirical_distribution_trajectory,
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
