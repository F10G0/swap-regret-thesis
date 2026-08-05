import numpy as np
import pytest

from metrics.equilibrium_convergence import (
    EquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
)


def test_distance_aggregation_preserves_horizons_and_replicate_mean() -> None:
    horizons = np.asarray([1, 10, 100])
    trajectories = [
        EquilibriumDistanceTrajectory(
            horizons,
            np.asarray([1.0, 0.5, 0.25]),
            np.asarray([0.8, 0.4, 0.2]),
        ),
        EquilibriumDistanceTrajectory(
            horizons.copy(),
            np.asarray([0.8, 0.3, 0.15]),
            np.asarray([0.6, 0.2, 0.1]),
        ),
    ]

    result = aggregate_equilibrium_distance_trajectories(trajectories)

    assert np.array_equal(result.horizons, horizons)
    assert np.allclose(result.ce_mean, [0.9, 0.4, 0.2])
    assert np.allclose(result.cce_mean, [0.7, 0.3, 0.15])
    assert result.n_replicates == 2
    assert np.all(result.ce_confidence > 0.0)
    assert np.all(result.cce_confidence > 0.0)


def test_distance_aggregation_rejects_mismatched_horizons() -> None:
    first = EquilibriumDistanceTrajectory(
        np.asarray([1, 10]),
        np.zeros(2),
        np.zeros(2),
    )
    second = EquilibriumDistanceTrajectory(
        np.asarray([1, 20]),
        np.zeros(2),
        np.zeros(2),
    )

    with pytest.raises(ValueError, match="matching horizons"):
        aggregate_equilibrium_distance_trajectories([first, second])


def test_distance_aggregation_requires_a_replicate() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_equilibrium_distance_trajectories([])
