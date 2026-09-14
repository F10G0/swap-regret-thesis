import numpy as np
import pytest

from metrics.equilibrium_distance import (
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


@pytest.mark.parametrize("n_players", [3, 4])
def test_full_space_convergence_supports_multiple_players(n_players):
    from metrics.empirical_distribution import empirical_distribution_trajectory
    from metrics.equilibrium_distance import equilibrium_distance_trajectory

    action_shape = (2,) * n_players
    payoffs = np.zeros((n_players, *action_shape))
    empirical = empirical_distribution_trajectory(
        [(0,) * n_players, (1,) * n_players], action_shape, checkpoints=[1, 2],
    )
    distances = equilibrium_distance_trajectory(payoffs, empirical)
    np.testing.assert_array_equal(distances.horizons, [1, 2])
    np.testing.assert_array_equal(distances.ce, [0.0, 0.0])
    np.testing.assert_array_equal(distances.cce, [0.0, 0.0])


def test_trajectory_prepares_once_per_concept_and_uses_only_scalar_objectives(monkeypatch):
    from types import SimpleNamespace
    import metrics.equilibrium_distance as module
    from metrics.empirical_distribution import EmpiricalDistributionTrajectory
    concepts, calls = [], []

    class Prepared:
        action_shape = (2, 2)

        def __init__(self, payoff_tensor, equilibrium):
            concepts.append(equilibrium)
            self.equilibrium = equilibrium

        def solve(self, vector):
            calls.append(self.equilibrium)
            return SimpleNamespace(fun=float(vector[0]))  # Deliberately no x.

    monkeypatch.setattr(module, "_PreparedDistanceLP", Prepared)
    empirical = EmpiricalDistributionTrajectory((2, 2), np.array([1, 10, 100]), np.full((3, 4), .25))
    result = module.equilibrium_distance_trajectory(np.zeros((2, 2, 2)), empirical)
    assert concepts == ["ce", "cce"]
    assert calls == ["ce", "cce"] * 3
    np.testing.assert_array_equal(result.ce, [.25] * 3)
