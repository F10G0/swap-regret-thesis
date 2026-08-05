import numpy as np
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from experiments.games import (
    create_matching_pennies_payoffs,
    create_rock_paper_scissors_payoffs,
)
from metrics.empirical_distribution import (
    EmpiricalDistributionTrajectory,
    empirical_distribution_trajectory,
    mean_empirical_distribution_trajectory,
)
from metrics.equilibrium_convergence import (
    EquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
)
from experimental.equilibrium_trajectory.analysis import (
    analyze_equilibrium_convergence,
    project_equilibrium_trajectory_comparison,
    project_equilibrium_trajectory,
    project_geometry_trajectory,
    project_unified_equilibrium_trajectory_comparison,
)
from metrics.equilibrium_distance import equilibrium_l1_distance
from experimental.equilibrium_trajectory.geometry import (
    analyze_equilibrium_projection_geometry,
)
from experimental.equilibrium_trajectory.projection import (
    _SupportResult,
    _reconstruct_projected_polygon,
    fit_equilibrium_comparison_projection,
    fit_equilibrium_projection,
    fit_unified_equilibrium_comparison_direction,
)


def coordination_game() -> np.ndarray:
    payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.stack((payoffs, payoffs))


def _sample_rps_trajectory(count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(17)
    samples = rng.dirichlet(np.ones(9), size=count)
    return samples


def test_rps_projection_has_point_ce_and_line_cce() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = empirical_distribution_trajectory(
        [(index % 3, (2 * index) % 3) for index in range(30)],
        (3, 3),
        checkpoints=[1, 10, 20, 30],
    )

    analysis = project_equilibrium_trajectory(
        payoffs,
        empirical,
        support_query_cap=12,
        geometry=geometry,
    )

    assert geometry.projection_case == "nested_lower_dimensional"
    assert analysis.view_kind == "equilibrium_relative"
    assert analysis.axis_labels == (
        "CE-relative CCE tangent coordinate",
        "L1 distance to CCE",
    )
    expected_x = (
        empirical.vectors - geometry.ce.reference
    ) @ analysis.projection.components[0]
    expected_y = np.asarray([
        equilibrium_l1_distance(
            payoffs,
            distribution,
            "cce",
        ).distance
        for distribution in empirical.distributions
    ])
    assert np.allclose(analysis.projected_trajectory[:, 0], expected_x)
    assert np.allclose(analysis.projected_trajectory[:, 1], expected_y)
    assert np.all(analysis.projected_trajectory[:, 1] >= 0.0)
    assert analysis.ce_region.affine_dimension == 0
    assert analysis.ce_region.certified
    assert analysis.ce_region.certification_mode == "exact"
    assert analysis.ce_region.support_query_count == 0
    assert analysis.ce_region.support_lp_count == 0
    assert analysis.ce_region.boundary.shape == (1, 2)
    assert np.allclose(analysis.ce_region.boundary[0], [0.0, 0.0])
    assert np.allclose(analysis.ce_region.support_points, 0.0)
    assert analysis.cce_region.affine_dimension == 1
    assert analysis.cce_region.certified
    assert analysis.cce_region.certification_mode == "exact"
    assert analysis.cce_region.support_query_count == 2
    assert analysis.cce_region.support_lp_count == 2
    assert analysis.cce_region.boundary.shape == (2, 2)
    normal_coordinates = analysis.cce_region.support_points[:, 1]
    assert np.allclose(
        normal_coordinates,
        0.0,
        atol=1e-8,
    )
    for distribution in analysis.cce_region.support_distributions:
        assert equilibrium_l1_distance(
            payoffs,
            distribution,
            "cce",
        ).distance == pytest.approx(
            0.0,
            abs=EQUILIBRIUM_LP_TOLERANCE,
        )


def test_rps_equilibrium_relative_view_preserves_geometry_line_endpoints() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([1, 2, 3]),
        _sample_rps_trajectory(3),
    )

    geometry_view = project_geometry_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
    )
    relative_view = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
    )

    assert np.allclose(
        relative_view.cce_region.boundary[:, 0],
        geometry_view.cce_region.boundary[:, 0],
    )
    assert np.allclose(relative_view.cce_region.boundary[:, 1], 0.0)


def test_exact_rps_cce_distribution_has_zero_relative_height() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([1]),
        geometry.cce.reference.reshape(1, -1),
    )

    analysis = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
    )

    assert analysis.projected_trajectory[0, 1] == pytest.approx(
        0.0,
        abs=EQUILIBRIUM_LP_TOLERANCE,
    )


def test_replicate_mean_relative_height_is_distance_of_mean_distribution() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    first_vector = np.zeros(9)
    first_vector[0] = 1.0
    second_vector = np.zeros(9)
    second_vector[4] = 1.0
    first = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([1]),
        first_vector.reshape(1, -1),
    )
    second = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([1]),
        second_vector.reshape(1, -1),
    )
    empirical_mean = mean_empirical_distribution_trajectory(
        [first, second]
    )

    analysis = project_equilibrium_trajectory(
        payoffs,
        empirical_mean,
        geometry=geometry,
    )
    distance_of_mean = equilibrium_l1_distance(
        payoffs,
        empirical_mean.distributions[0],
        "cce",
    ).distance
    mean_distance = np.mean([
        equilibrium_l1_distance(
            payoffs,
            trajectory.distributions[0],
            "cce",
        ).distance
        for trajectory in (first, second)
    ])

    assert analysis.projected_trajectory[0, 1] == pytest.approx(
        distance_of_mean
    )
    assert distance_of_mean != pytest.approx(mean_distance)


def test_focus_fit_excludes_preceding_log_node_but_still_transforms_it() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    vectors = np.vstack((
        np.eye(1, 9, 0)[0],
        _sample_rps_trajectory(4),
    ))
    empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([100, 1_000, 3_250, 5_500, 10_000]),
        vectors,
    )
    focused_empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        empirical.horizons[1:],
        empirical.vectors[1:],
    )

    analysis = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
        fit_empirical=focused_empirical,
    )
    expected_projection = fit_equilibrium_projection(
        geometry,
        [focused_empirical.vectors],
    )

    assert np.allclose(
        analysis.projection.components,
        expected_projection.components,
    )
    assert len(analysis.projected_trajectory) == len(empirical.horizons)
    assert analysis.projected_trajectory[0, 0] == pytest.approx(
        expected_projection.transform(empirical.vectors[:1])[0, 0]
    )
    assert analysis.projected_trajectory[0, 1] == pytest.approx(
        equilibrium_l1_distance(
            payoffs,
            empirical.distributions[0],
            "cce",
        ).distance
    )


def test_one_member_comparison_preserves_projection_and_ce_reference() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    samples = _sample_rps_trajectory(7)

    original = fit_equilibrium_projection(geometry, [samples])
    comparison = fit_equilibrium_comparison_projection(
        geometry,
        [samples],
    )

    assert np.allclose(comparison.components, original.components)
    assert np.array_equal(comparison.center, geometry.ce.reference)
    assert np.array_equal(original.center, geometry.ce.reference)


def test_comparison_projection_is_order_independent_and_scale_normalized() -> None:
    geometry = analyze_equilibrium_projection_geometry(coordination_game())
    reference = geometry.ce.reference
    first_direction = geometry.x_subspace_basis[:, 0]
    second_direction = geometry.x_subspace_basis[:, 1]
    path_coefficients = np.array([-1.0, -0.2, 1.2])[:, None]

    def trajectories(path_scale: float, separation_scale: float):
        common = reference + path_scale * path_coefficients * first_direction
        separation = separation_scale * second_direction
        return [common + separation, common - separation]

    baseline = trajectories(1.0, 1.0)
    rescaled = trajectories(100.0, 0.01)
    first = fit_equilibrium_comparison_projection(geometry, baseline)
    scaled = fit_equilibrium_comparison_projection(geometry, rescaled)
    reversed_members = fit_equilibrium_comparison_projection(
        geometry,
        list(reversed(baseline)),
    )

    assert np.allclose(first.components, scaled.components, atol=1e-10)
    assert np.allclose(
        first.components,
        reversed_members.components,
        atol=1e-10,
    )
    assert np.array_equal(first.center, geometry.ce.reference)


def test_rps_comparison_shares_projection_and_true_cce_distance_axis() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    horizons = np.array([1, 10, 100, 1_000])
    first = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        _sample_rps_trajectory(4),
    )
    second = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        np.flip(_sample_rps_trajectory(4), axis=0).copy(),
    )

    comparison = project_equilibrium_trajectory_comparison(
        payoffs,
        [first, second],
        member_ids=["first", "second"],
        geometry=geometry,
    )

    assert comparison.view_kind == "equilibrium_relative"
    assert np.array_equal(comparison.projection.center, geometry.ce.reference)
    assert np.allclose(comparison.ce_region.boundary, [[0.0, 0.0]])
    assert np.allclose(comparison.cce_region.boundary[:, 1], 0.0)
    for member, empirical in zip(comparison.members, (first, second)):
        expected_x = (
            empirical.vectors - geometry.ce.reference
        ) @ comparison.projection.components[0]
        expected_y = np.asarray([
            equilibrium_l1_distance(payoffs, distribution, "cce").distance
            for distribution in empirical.distributions
        ])
        assert np.allclose(member.projected_trajectory[:, 0], expected_x)
        assert np.allclose(member.projected_trajectory[:, 1], expected_y)


def test_rps_comparison_renders_regions_once_and_measures_each_member_checkpoint(
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import analysis as convergence_module
    from experimental.equilibrium_trajectory import projection as projection_module

    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    horizons = np.array([1, 10, 100])
    empiricals = [
        EmpiricalDistributionTrajectory(
            (3, 3),
            horizons,
            np.roll(_sample_rps_trajectory(3), shift, axis=1),
        )
        for shift in (0, 1, 2)
    ]
    support_calls = []
    distance_calls = []
    original_support = projection_module.optimize_equilibrium
    original_distance = convergence_module.equilibrium_l1_distance

    def counted_support(payoff_tensor, equilibrium, objective):
        support_calls.append(equilibrium)
        return original_support(payoff_tensor, equilibrium, objective)

    def counted_distance(payoff_tensor, distribution, equilibrium):
        distance_calls.append(equilibrium)
        return original_distance(payoff_tensor, distribution, equilibrium)

    monkeypatch.setattr(
        projection_module,
        "optimize_equilibrium",
        counted_support,
    )
    monkeypatch.setattr(
        convergence_module,
        "equilibrium_l1_distance",
        counted_distance,
    )

    comparison = project_equilibrium_trajectory_comparison(
        payoffs,
        empiricals,
        member_ids=["a", "b", "c"],
        geometry=geometry,
    )

    assert support_calls == ["cce", "cce"]
    assert distance_calls == ["cce"] * (len(empiricals) * len(horizons))
    assert comparison.ce_region.support_lp_count == 0
    assert comparison.cce_region.support_lp_count == 2


def test_unified_comparison_uses_one_simplex_tangent_direction_and_true_cce_distance() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    horizons = np.array([1, 10, 100, 1_000])
    first = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        _sample_rps_trajectory(4),
    )
    second = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        np.roll(_sample_rps_trajectory(4), 1, axis=1),
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [first, second],
        member_ids=["first", "second"],
    )
    direction = comparison.projection.components[0]
    x_center = float(comparison.projection.center @ direction)

    assert comparison.view_kind == "unified_equilibrium_relative"
    assert np.sum(direction) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert np.allclose(comparison.projection.components[1], 0.0)
    assert np.mean(comparison.ce_region.boundary[:, 0]) == pytest.approx(
        0.0,
        abs=EQUILIBRIUM_LP_TOLERANCE,
    )
    assert (
        comparison.cce_region.boundary[0, 0]
        <= comparison.ce_region.boundary[0, 0]
        + EQUILIBRIUM_LP_TOLERANCE
    )
    assert (
        comparison.cce_region.boundary[-1, 0]
        >= comparison.ce_region.boundary[-1, 0]
        - EQUILIBRIUM_LP_TOLERANCE
    )
    for region in (comparison.ce_region, comparison.cce_region):
        expected_x = (
            region.support_distributions.reshape(2, -1) @ direction
            - x_center
        )
        assert np.allclose(region.support_points[:, 0], expected_x)
        assert np.allclose(region.support_points[:, 1], 0.0)
        assert region.support_query_count == 2
        assert region.support_lp_count == 2
    for member, empirical in zip(comparison.members, (first, second)):
        expected_y = np.asarray([
            equilibrium_l1_distance(payoffs, distribution, "cce").distance
            for distribution in empirical.distributions
        ])
        assert np.allclose(
            member.projected_trajectory[:, 0],
            empirical.vectors @ direction - x_center,
        )
        assert np.allclose(member.projected_trajectory[:, 1], expected_y)
        assert np.all(member.projected_trajectory[:, 1] >= 0.0)


def test_unified_comparison_centers_a_positive_width_ce_interval() -> None:
    payoffs = np.zeros((2, 2, 2), dtype=float)
    horizons = np.array([1, 10, 100])
    first = EmpiricalDistributionTrajectory(
        (2, 2),
        horizons,
        np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]),
    )
    second = EmpiricalDistributionTrajectory(
        (2, 2),
        horizons,
        np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]),
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [first, second],
        member_ids=["first", "second"],
    )

    assert comparison.ce_region.affine_dimension == 1
    assert comparison.cce_region.affine_dimension == 1
    assert comparison.ce_region.boundary.shape == (2, 2)
    assert np.mean(comparison.ce_region.boundary[:, 0]) == pytest.approx(0.0)
    assert np.allclose(
        comparison.ce_region.boundary,
        comparison.cce_region.boundary,
    )
    for member in comparison.members:
        assert np.allclose(member.projected_trajectory[:, 1], 0.0)


def test_unified_comparison_is_member_order_independent() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    horizons = np.array([1, 10, 100])
    first = EmpiricalDistributionTrajectory(
        (3, 3), horizons, _sample_rps_trajectory(3)
    )
    second = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        np.roll(_sample_rps_trajectory(3), 2, axis=1),
    )

    forward = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [first, second],
        member_ids=["first", "second"],
    )
    reverse = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [second, first],
        member_ids=["second", "first"],
    )

    assert np.allclose(
        forward.projection.components,
        reverse.projection.components,
    )
    assert np.allclose(
        forward.projection.center,
        reverse.projection.center,
    )
    forward_members = {
        member.member_id: member.projected_trajectory
        for member in forward.members
    }
    reverse_members = {
        member.member_id: member.projected_trajectory
        for member in reverse.members
    }
    assert forward_members.keys() == reverse_members.keys()
    for member_id in forward_members:
        assert np.allclose(
            forward_members[member_id],
            reverse_members[member_id],
        )


def test_one_member_unified_comparison_and_exact_cce_height() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    horizons = np.array([1, 10])
    empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        horizons,
        np.vstack((
            _sample_rps_trajectory(1),
            geometry.cce.reference,
        )),
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [empirical],
        member_ids=["only"],
    )

    assert len(comparison.members) == 1
    assert np.linalg.norm(comparison.projection.components[0]) == pytest.approx(1.0)
    assert comparison.members[0].projected_trajectory[-1, 1] == pytest.approx(
        0.0,
        abs=EQUILIBRIUM_LP_TOLERANCE,
    )


def test_unified_projection_does_not_use_affine_geometry_or_adaptive_regions(
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import analysis as convergence_module

    payoffs = create_rock_paper_scissors_payoffs()
    empirical = EmpiricalDistributionTrajectory(
        (3, 3),
        np.array([1, 10]),
        _sample_rps_trajectory(2),
    )
    monkeypatch.setattr(
        convergence_module,
        "analyze_equilibrium_projection_geometry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("affine geometry must not be requested")
        ),
    )
    monkeypatch.setattr(
        convergence_module,
        "project_equilibrium_set",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("adaptive region projection must not be requested")
        ),
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [empirical],
    )

    assert comparison.ce_region.support_lp_count == 2
    assert comparison.cce_region.support_lp_count == 2


def test_unified_comparison_uses_four_support_lps_and_one_distance_lp_per_node(
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import analysis as convergence_module

    payoffs = create_rock_paper_scissors_payoffs()
    horizons = np.array([1, 10, 100])
    empiricals = [
        EmpiricalDistributionTrajectory(
            (3, 3),
            horizons,
            np.roll(_sample_rps_trajectory(3), shift, axis=1),
        )
        for shift in (0, 1)
    ]
    support_calls = []
    distance_calls = []
    original_support = convergence_module.optimize_equilibrium
    original_distance = convergence_module.equilibrium_l1_distance

    def counted_support(payoff_tensor, equilibrium, objective):
        support_calls.append((equilibrium, np.asarray(objective).copy()))
        return original_support(payoff_tensor, equilibrium, objective)

    def counted_distance(payoff_tensor, distribution, equilibrium):
        distance_calls.append(equilibrium)
        return original_distance(payoff_tensor, distribution, equilibrium)

    monkeypatch.setattr(
        convergence_module,
        "optimize_equilibrium",
        counted_support,
    )
    monkeypatch.setattr(
        convergence_module,
        "equilibrium_l1_distance",
        counted_distance,
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        empiricals,
    )
    direction = comparison.projection.components[0].reshape(3, 3)

    assert [equilibrium for equilibrium, _ in support_calls] == [
        "ce", "ce", "cce", "cce"
    ]
    assert np.allclose(support_calls[0][1], -direction)
    assert np.allclose(support_calls[1][1], direction)
    assert np.allclose(support_calls[2][1], -direction)
    assert np.allclose(support_calls[3][1], direction)
    assert distance_calls == ["cce"] * (
        len(empiricals) * len(horizons)
    )


def test_unified_single_profile_game_renders_degenerate_intervals() -> None:
    payoffs = np.zeros((2, 1, 1))
    empirical = EmpiricalDistributionTrajectory(
        (1, 1),
        np.array([1]),
        np.ones((1, 1)),
    )

    comparison = project_unified_equilibrium_trajectory_comparison(
        payoffs,
        [empirical],
    )

    assert np.allclose(comparison.projection.components, 0.0)
    assert comparison.ce_region.affine_dimension == 0
    assert comparison.cce_region.affine_dimension == 0
    assert np.allclose(comparison.ce_region.boundary, [[0.0, 0.0]])
    assert np.allclose(comparison.cce_region.boundary, [[0.0, 0.0]])


def test_unified_direction_uses_balanced_comparison_objective() -> None:
    samples = _sample_rps_trajectory(3)
    first = [samples, np.roll(samples, 1, axis=1)]
    rescaled = [
        np.mean(first, axis=0)
        + 100.0 * (trajectory - np.mean(first, axis=0))
        for trajectory in first
    ]

    baseline = fit_unified_equilibrium_comparison_direction(first)
    separation_rescaled = fit_unified_equilibrium_comparison_direction(
        rescaled
    )

    assert np.allclose(baseline, separation_rescaled, atol=1e-10)
    assert np.sum(baseline) == pytest.approx(0.0, abs=1e-12)


def test_constrained_rps_axes_use_required_subspaces_and_maximize_variance() -> None:
    geometry = analyze_equilibrium_projection_geometry(
        create_rock_paper_scissors_payoffs()
    )
    samples = _sample_rps_trajectory()
    projection = fit_equilibrium_projection(geometry, [samples])
    x_axis, y_axis = projection.components

    assert np.linalg.norm(x_axis) == pytest.approx(1.0)
    assert np.linalg.norm(y_axis) == pytest.approx(1.0)
    assert np.dot(x_axis, y_axis) == pytest.approx(0.0, abs=1e-10)
    assert np.linalg.norm(
        x_axis
        - geometry.x_subspace_basis
        @ (geometry.x_subspace_basis.T @ x_axis)
    ) < 1e-9
    assert np.linalg.norm(
        y_axis
        - geometry.y_subspace_basis
        @ (geometry.y_subspace_basis.T @ y_axis)
    ) < 1e-9

    centered = samples - np.mean(samples, axis=0)
    for axis, basis in (
        (x_axis, geometry.x_subspace_basis),
        (y_axis, geometry.y_subspace_basis),
    ):
        allowed = centered @ basis
        expected_variance = np.linalg.eigvalsh(
            allowed.T @ allowed
        )[-1]
        actual_variance = np.linalg.norm(centered @ axis) ** 2
        assert actual_variance == pytest.approx(expected_variance)


def test_rps_point_and_line_rendering_uses_only_two_support_lps(
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import projection as projection_module

    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = empirical_distribution_trajectory(
        [(index % 3, (index + 1) % 3) for index in range(12)],
        (3, 3),
        checkpoints=[1, 4, 8, 12],
    )
    calls = []
    original = projection_module.optimize_equilibrium

    def counted(payoff_tensor, equilibrium, objective):
        calls.append(equilibrium)
        return original(payoff_tensor, equilibrium, objective)

    monkeypatch.setattr(
        projection_module,
        "optimize_equilibrium",
        counted,
    )
    project_equilibrium_trajectory(
        payoffs,
        empirical,
        support_query_cap=12,
        geometry=geometry,
    )

    assert calls == ["cce", "cce"]


def test_rps_relative_trajectory_adds_one_cce_distance_lp_per_checkpoint(
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import analysis as convergence_module

    payoffs = create_rock_paper_scissors_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = empirical_distribution_trajectory(
        [(index % 3, (index + 1) % 3) for index in range(12)],
        (3, 3),
        checkpoints=[1, 4, 8, 12],
    )
    calls = []
    original = convergence_module.equilibrium_l1_distance

    def counted(payoff_tensor, distribution, equilibrium):
        calls.append((np.asarray(distribution).copy(), equilibrium))
        return original(payoff_tensor, distribution, equilibrium)

    monkeypatch.setattr(
        convergence_module,
        "equilibrium_l1_distance",
        counted,
    )
    project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
    )

    assert len(calls) == len(empirical.horizons)
    assert [equilibrium for _, equilibrium in calls] == ["cce"] * 4
    assert all(
        np.allclose(distribution, expected)
        for (distribution, _), expected in zip(
            calls,
            empirical.distributions,
        )
    )


def test_rank_deficient_constrained_projection_is_deterministic() -> None:
    payoffs = create_matching_pennies_payoffs()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    samples = np.repeat(
        geometry.ce.reference.reshape(1, -1),
        repeats=5,
        axis=0,
    )

    first = fit_equilibrium_projection(geometry, [samples])
    second = fit_equilibrium_projection(geometry, [samples])

    assert np.array_equal(first.components, second.components)
    assert np.allclose(
        first.components @ first.components.T,
        np.eye(2),
    )


def test_full_dimensional_equilibria_keep_two_dimensional_regions() -> None:
    payoffs = coordination_game()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = empirical_distribution_trajectory(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        (2, 2),
        checkpoints=[1, 2, 3, 4],
    )

    analysis = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
    )

    assert geometry.projection_case == "full_dimensional"
    assert analysis.view_kind == "geometry"
    assert np.allclose(
        analysis.projected_trajectory,
        analysis.projection.transform(empirical.vectors),
    )
    assert analysis.ce_region.affine_dimension == 2
    assert analysis.cce_region.affine_dimension == 2
    assert analysis.ce_region.certified
    assert analysis.cce_region.certified
    assert analysis.ce_region.support_query_count < 128
    assert analysis.cce_region.support_query_count < 128


@pytest.mark.parametrize(
    ("action_shape", "actions"),
    [
        ((2, 1, 2), [(0, 0, 0), (1, 0, 1)]),
        ((1, 2, 1, 2), [(0, 0, 0, 0), (0, 1, 0, 1)]),
    ],
)
def test_convergence_analysis_supports_three_and_four_players(
    action_shape,
    actions,
) -> None:
    payoff_tensor = np.zeros((len(action_shape), *action_shape))
    empirical = empirical_distribution_trajectory(
        actions,
        action_shape,
        checkpoints=[1, 2],
    )
    analysis = analyze_equilibrium_convergence(
        payoff_tensor,
        empirical,
        support_query_cap=32,
    )

    assert analysis.projected_trajectory.shape == (2, 2)
    assert np.allclose(
        analysis.distances.ce,
        0.0,
        atol=EQUILIBRIUM_LP_TOLERANCE,
    )
    assert np.allclose(
        analysis.distances.cce,
        0.0,
        atol=EQUILIBRIUM_LP_TOLERANCE,
    )


def test_equilibrium_distances_are_averaged_after_each_replicate_is_measured() -> None:
    first = EquilibriumDistanceTrajectory(
        np.array([10, 100]),
        np.array([0.1, 0.2]),
        np.array([0.0, 0.1]),
    )
    second = EquilibriumDistanceTrajectory(
        np.array([10, 100]),
        np.array([0.3, 0.4]),
        np.array([0.2, 0.3]),
    )

    aggregate = aggregate_equilibrium_distance_trajectories(
        [first, second]
    )

    assert np.allclose(aggregate.ce_mean, [0.2, 0.3])
    assert np.allclose(aggregate.cce_mean, [0.1, 0.2])
    assert np.allclose(
        aggregate.ce_confidence,
        [1.2706204736, 1.2706204736],
    )
    assert np.allclose(
        aggregate.cce_confidence,
        [1.2706204736, 1.2706204736],
    )


class _SyntheticPolygonOracle:
    def __init__(
        self,
        vertices: np.ndarray,
        use_face_midpoints: bool = True,
    ):
        self.vertices = np.asarray(vertices, dtype=float)
        self.use_face_midpoints = use_face_midpoints
        self.lp_solve_count = 0
        self.face_call_count = 0
        self.support_directions: list[np.ndarray] = []

    def support(self, direction: np.ndarray) -> _SupportResult:
        self.lp_solve_count += 1
        self.support_directions.append(direction.copy())
        values = self.vertices @ direction
        support = float(np.max(values))
        exposed = self.vertices[
            np.isclose(values, support, atol=1e-12)
        ]
        if self.use_face_midpoints:
            point = np.mean(exposed, axis=0)
        else:
            direction_key = tuple(np.round(direction, decimals=12))
            corners = {
                (1.0, 0.0): np.array([2.0, -1.0]),
                (-1.0, 0.0): np.array([-2.0, 1.0]),
                (0.0, 1.0): np.array([2.0, 1.0]),
                (0.0, -1.0): np.array([-2.0, -1.0]),
            }
            point = corners.get(direction_key, exposed[0])
        return _SupportResult(
            support,
            point,
            point.copy(),
        )

    def support_extreme_point(
        self,
        direction: np.ndarray,
        primary: _SupportResult,
        orientation: float,
    ) -> _SupportResult:
        del primary
        self.face_call_count += 1
        self.lp_solve_count += 1
        values = self.vertices @ direction
        support = float(np.max(values))
        exposed = self.vertices[
            np.isclose(values, support, atol=1e-12)
        ]
        perpendicular = np.array([-direction[1], direction[0]])
        values = exposed @ perpendicular
        point = exposed[
            int(np.argmax(values) if orientation >= 0.0 else np.argmin(values))
        ]
        return _SupportResult(
            support,
            point,
            point.copy(),
        )


def _polygon_points(result) -> np.ndarray:
    return np.asarray([vertex.point for vertex in result.vertices])


def _maximum_support_gap(
    boundary: np.ndarray,
    exact_vertices: np.ndarray,
) -> float:
    gaps = []
    for position, first in enumerate(boundary):
        second = boundary[(position + 1) % len(boundary)]
        edge = second - first
        normal = np.array([edge[1], -edge[0]])
        normal /= np.linalg.norm(normal)
        gaps.append(
            float(np.max(exact_vertices @ normal) - first @ normal)
        )
    return max(gaps, default=0.0)


def test_adaptive_rectangle_recovers_exposed_face_endpoints() -> None:
    vertices = np.array([
        [-2.0, -1.0],
        [2.0, -1.0],
        [2.0, 1.0],
        [-2.0, 1.0],
    ])
    oracle = _SyntheticPolygonOracle(vertices)

    result = _reconstruct_projected_polygon(
        oracle.support,
        oracle.support_extreme_point,
        support_query_cap=64,
        tolerance=1e-9,
        label="rectangle",
    )

    assert result.certified
    assert np.allclose(_polygon_points(result), vertices)
    assert result.support_query_count == 8
    assert oracle.face_call_count == 4
    assert oracle.lp_solve_count == 12
    assert oracle.lp_solve_count < 16
    assert result.support_query_count < 128


def test_certified_edges_do_not_trigger_secondary_optimization() -> None:
    vertices = np.array([
        [-2.0, -1.0],
        [2.0, -1.0],
        [2.0, 1.0],
        [-2.0, 1.0],
    ])
    oracle = _SyntheticPolygonOracle(
        vertices,
        use_face_midpoints=False,
    )

    result = _reconstruct_projected_polygon(
        oracle.support,
        oracle.support_extreme_point,
        support_query_cap=64,
        tolerance=1e-9,
        label="rectangle",
    )

    assert result.certified
    assert np.allclose(_polygon_points(result), vertices)
    assert result.support_query_count == 4
    assert oracle.face_call_count == 0
    assert oracle.lp_solve_count == 4


class _DiagonalRectangleOracle(_SyntheticPolygonOracle):
    def support(self, direction: np.ndarray) -> _SupportResult:
        self.lp_solve_count += 1
        self.support_directions.append(direction.copy())
        values = self.vertices @ direction
        support = float(np.max(values))
        exposed = self.vertices[
            np.isclose(values, support, atol=1e-12)
        ]
        diagonal = exposed @ np.ones(2)
        point = (
            exposed[int(np.argmax(diagonal))]
            if float(np.sum(direction)) > 0.0
            else exposed[int(np.argmin(diagonal))]
        )
        return _SupportResult(
            support,
            point,
            point.copy(),
        )


def test_degenerate_initial_hull_recovers_nonunique_face_endpoints() -> None:
    vertices = np.array([
        [-2.0, -1.0],
        [2.0, -1.0],
        [2.0, 1.0],
        [-2.0, 1.0],
    ])
    oracle = _DiagonalRectangleOracle(vertices)

    result = _reconstruct_projected_polygon(
        oracle.support,
        oracle.support_extreme_point,
        support_query_cap=64,
        tolerance=1e-9,
        label="rectangle",
    )

    assert result.certified
    assert np.allclose(_polygon_points(result), vertices)
    assert oracle.face_call_count >= 1
    assert sum(
        np.allclose(direction, [1.0, 0.0])
        for direction in oracle.support_directions
    ) == 1


def test_adaptive_triangle_is_deterministic() -> None:
    vertices = np.array([
        [-1.0, -1.0],
        [2.0, -1.0],
        [0.0, 2.0],
    ])

    results = []
    for _ in range(2):
        oracle = _SyntheticPolygonOracle(vertices)
        results.append(
            _reconstruct_projected_polygon(
                oracle.support,
                oracle.support_extreme_point,
                support_query_cap=64,
                tolerance=1e-9,
                label="triangle",
            )
        )

    assert results[0].certified
    assert results[1].certified
    assert np.array_equal(
        _polygon_points(results[0]),
        _polygon_points(results[1]),
    )
    assert np.allclose(_polygon_points(results[0]), vertices)
    assert results[0].support_query_count < 128


def test_adaptive_polygon_warns_and_returns_inner_hull_at_cap() -> None:
    vertices = np.array([
        [-2.0, -1.0],
        [2.0, -1.0],
        [2.0, 1.0],
        [-2.0, 1.0],
    ])
    oracle = _SyntheticPolygonOracle(vertices)

    with pytest.warns(RuntimeWarning, match="support-query cap"):
        result = _reconstruct_projected_polygon(
            oracle.support,
            oracle.support_extreme_point,
            support_query_cap=4,
            tolerance=1e-9,
            label="rectangle",
        )

    assert not result.certified
    assert result.certification_mode == "safety_cap"
    assert result.support_query_count == 4
    assert oracle.lp_solve_count == 4


def test_render_tolerance_skips_a_visually_small_vertex() -> None:
    vertices = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.51, 0.51],
        [0.0, 1.0],
    ])
    render_oracle = _SyntheticPolygonOracle(vertices)
    exact_oracle = _SyntheticPolygonOracle(vertices)

    rendered = _reconstruct_projected_polygon(
        render_oracle.support,
        render_oracle.support_extreme_point,
        support_query_cap=64,
        tolerance=1e-9,
        label="small-feature polygon",
        relative_render_tolerance=1e-2,
    )
    exact = _reconstruct_projected_polygon(
        exact_oracle.support,
        exact_oracle.support_extreme_point,
        support_query_cap=64,
        tolerance=1e-9,
        label="small-feature polygon",
        relative_render_tolerance=None,
    )

    assert rendered.certified
    assert rendered.certification_mode == "render"
    assert len(rendered.vertices) == 4
    assert exact.certified
    assert exact.certification_mode == "exact"
    assert len(exact.vertices) == 5
    assert rendered.support_query_count < exact.support_query_count
    observed_gap = _maximum_support_gap(
        _polygon_points(rendered),
        vertices,
    )
    assert observed_gap <= rendered.support_gap_tolerance
    assert rendered.max_observed_support_gap == pytest.approx(
        observed_gap
    )


def test_tightening_render_tolerance_monotonically_approaches_exact() -> None:
    vertices = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.51, 0.51],
        [0.0, 1.0],
    ])
    tolerances = [1e-2, 5e-3, 1e-3, None]
    vertex_counts = []
    errors = []

    for render_tolerance in tolerances:
        oracle = _SyntheticPolygonOracle(vertices)
        result = _reconstruct_projected_polygon(
            oracle.support,
            oracle.support_extreme_point,
            support_query_cap=64,
            tolerance=1e-9,
            label="small-feature polygon",
            relative_render_tolerance=render_tolerance,
        )
        assert result.certified
        boundary = _polygon_points(result)
        vertex_counts.append(len(boundary))
        errors.append(_maximum_support_gap(boundary, vertices))

    assert vertex_counts == sorted(vertex_counts)
    assert errors == sorted(errors, reverse=True)
    assert vertex_counts[-1] == len(vertices)
    assert errors[-1] == pytest.approx(0.0, abs=1e-12)


def test_region_tolerance_does_not_change_projection_or_trajectory() -> None:
    payoffs = coordination_game()
    geometry = analyze_equilibrium_projection_geometry(payoffs)
    empirical = empirical_distribution_trajectory(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        (2, 2),
        checkpoints=[1, 2, 3, 4],
    )

    rendered = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
        relative_render_tolerance=1e-2,
    )
    exact = project_equilibrium_trajectory(
        payoffs,
        empirical,
        geometry=geometry,
        relative_render_tolerance=None,
    )

    assert np.array_equal(
        rendered.projection.components,
        exact.projection.components,
    )
    assert np.array_equal(
        rendered.projected_trajectory,
        exact.projected_trajectory,
    )
    assert rendered.ce_region.certification_mode == "render"
    assert rendered.cce_region.certification_mode == "render"
    assert exact.ce_region.certification_mode == "exact"
    assert exact.cce_region.certification_mode == "exact"
