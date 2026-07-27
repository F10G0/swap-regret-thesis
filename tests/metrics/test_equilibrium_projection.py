import numpy as np
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.empirical_distribution import empirical_distribution_trajectory
from metrics.equilibrium_convergence import EquilibriumDistanceTrajectory, aggregate_equilibrium_distance_trajectories, analyze_equilibrium_convergence
from metrics.equilibrium_distance import equilibrium_l1_distance
from metrics.equilibrium_projection import LinearProjection2D, project_equilibrium_region


def coordination_game() -> np.ndarray:
    payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.stack((payoffs, payoffs))


def test_shared_projection_maps_the_same_distribution_deterministically() -> None:
    first = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    second = np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0]])
    projection = LinearProjection2D.fit([first, second])

    assert projection.transform(first).shape == (2, 2)
    assert np.allclose(projection.transform(first[1]), projection.transform(first)[1])
    assert np.allclose(projection.transform(second[0]), projection.transform(second)[0])


def test_projection_supports_three_player_distribution_vectors() -> None:
    distributions = np.zeros((3, 12))
    distributions[0, 0] = 1.0
    distributions[1, 5] = 1.0
    distributions[2, 11] = 1.0
    projection = LinearProjection2D.fit([distributions])

    assert projection.transform(distributions).shape == (3, 2)


@pytest.mark.parametrize("equilibrium", ["ce", "cce"])
def test_projected_support_distributions_are_upstream_equilibria(equilibrium: str) -> None:
    trajectory = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]])
    projection = LinearProjection2D.fit([trajectory])
    region = project_equilibrium_region(coordination_game(), projection, equilibrium, direction_count=12)

    assert region.support_points.shape[1] == 2
    for distribution in region.support_distributions:
        distance = equilibrium_l1_distance(coordination_game(), distribution, equilibrium).distance
        assert distance == pytest.approx(0.0, abs=EQUILIBRIUM_LP_TOLERANCE)


def test_rank_zero_and_rank_one_projected_regions_do_not_crash() -> None:
    point_projection = LinearProjection2D.fit([np.full((2, 4), 0.25)])
    point_region = project_equilibrium_region(coordination_game(), point_projection, "ce", direction_count=8)
    line_projection = LinearProjection2D.fit([np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])])
    line_region = project_equilibrium_region(coordination_game(), line_projection, "cce", direction_count=8)

    assert point_region.affine_dimension == 0
    assert point_region.boundary.shape == (1, 2)
    assert line_region.affine_dimension == 1
    assert line_region.boundary.shape[1] == 2


@pytest.mark.parametrize(
    ("action_shape", "actions"),
    [
        ((2, 1, 2), [(0, 0, 0), (1, 0, 1)]),
        ((1, 2, 1, 2), [(0, 0, 0, 0), (0, 1, 0, 1)]),
    ],
)
def test_convergence_analysis_supports_three_and_four_players(action_shape, actions) -> None:
    payoff_tensor = np.zeros((len(action_shape), *action_shape))
    empirical = empirical_distribution_trajectory(actions, action_shape, checkpoints=[1, 2])
    analysis = analyze_equilibrium_convergence(payoff_tensor, empirical, direction_count=8)

    assert analysis.projected_trajectory.shape == (2, 2)
    assert np.allclose(analysis.distances.ce, 0.0, atol=EQUILIBRIUM_LP_TOLERANCE)
    assert np.allclose(analysis.distances.cce, 0.0, atol=EQUILIBRIUM_LP_TOLERANCE)


def test_equilibrium_distances_are_averaged_after_each_replicate_is_measured() -> None:
    first = EquilibriumDistanceTrajectory(np.array([10, 100]), np.array([0.1, 0.2]), np.array([0.0, 0.1]))
    second = EquilibriumDistanceTrajectory(np.array([10, 100]), np.array([0.3, 0.4]), np.array([0.2, 0.3]))

    aggregate = aggregate_equilibrium_distance_trajectories([first, second])

    assert np.allclose(aggregate.ce_mean, [0.2, 0.3])
    assert np.allclose(aggregate.cce_mean, [0.1, 0.2])
    assert np.allclose(aggregate.ce_confidence, [0.196, 0.196])
    assert np.allclose(aggregate.cce_confidence, [0.196, 0.196])
