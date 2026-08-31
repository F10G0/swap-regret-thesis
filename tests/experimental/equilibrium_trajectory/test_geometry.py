from pathlib import Path

import numpy as np

from experiments.game_catalog import (
    load_game_payoffs,
    payoff_tensor_digest,
)
from experimental.equilibrium_trajectory import geometry as geometry_module
from experimental.equilibrium_trajectory.geometry import (
    EquilibriumAffineGeometry,
    EquilibriumGeometryCache,
    _projection_family,
    _simplex_tangent_basis,
    analyze_equilibrium_projection_geometry,
)
from tests.support import coordination_game_payoffs


def test_rps_affine_dimensions_and_projection_family_are_generic() -> None:
    geometry = analyze_equilibrium_projection_geometry(
        load_game_payoffs("rps")
    )

    assert geometry.simplex_dimension == 8
    assert geometry.ce.dimension == 0
    assert geometry.cce.dimension == 3
    assert geometry.projection_case == "nested_lower_dimensional"
    assert geometry.ce_projected_dimension == 0
    assert geometry.cce_projected_dimension == 1
    assert geometry.x_subspace_basis.shape == (9, 3)
    assert geometry.y_subspace_basis.shape == (9, 5)


def test_ce_direction_space_is_contained_in_cce_direction_space() -> None:
    geometries = [
        analyze_equilibrium_projection_geometry(
            load_game_payoffs(game)
        )
        for game in ("rps", "matching_pennies")
    ]
    geometries.append(
        analyze_equilibrium_projection_geometry(coordination_game_payoffs())
    )
    for geometry in geometries:
        if geometry.ce.dimension == 0:
            continue
        residual = (
            geometry.ce.direction_basis
            - geometry.cce.direction_basis
            @ (
                geometry.cce.direction_basis.T
                @ geometry.ce.direction_basis
            )
        )
        assert np.linalg.norm(residual) < 1e-8


def test_equal_dimension_point_fallback_for_matching_pennies() -> None:
    geometry = analyze_equilibrium_projection_geometry(
        load_game_payoffs("matching_pennies")
    )

    assert geometry.simplex_dimension == 3
    assert geometry.ce.dimension == 0
    assert geometry.cce.dimension == 0
    assert geometry.projection_case == "common_affine_hull_point"
    assert geometry.ce_projected_dimension == 0
    assert geometry.cce_projected_dimension == 0
    assert geometry.shared_axis_subspace


def test_full_dimensional_cce_fallback_remains_two_dimensional() -> None:
    geometry = analyze_equilibrium_projection_geometry(
        coordination_game_payoffs()
    )

    assert geometry.ce.dimension == geometry.simplex_dimension
    assert geometry.cce.dimension == geometry.simplex_dimension
    assert geometry.cce_projected_dimension == 2


def test_equal_codimension_one_and_full_cce_case_selection() -> None:
    action_shape = (2, 2)
    tangent = _simplex_tangent_basis(4)
    reference = np.full(4, 0.25)
    common = EquilibriumAffineGeometry(
        reference,
        tangent[:, :2],
        tangent[:, 2:],
    )
    common_geometry = _projection_family(
        action_shape,
        common,
        common,
    )

    assert common_geometry.projection_case == "common_affine_hull_line"
    assert common_geometry.ce_projected_dimension == 1
    assert common_geometry.cce_projected_dimension == 1

    ce = EquilibriumAffineGeometry(
        reference,
        tangent[:, :1],
        tangent[:, 1:],
    )
    cce = EquilibriumAffineGeometry(
        reference,
        tangent,
        np.zeros((4, 0)),
    )
    full_cce_geometry = _projection_family(action_shape, ce, cce)

    assert full_cce_geometry.projection_case == "full_cce_point_ce"
    assert full_cce_geometry.ce_projected_dimension == 0
    assert full_cce_geometry.cce_projected_dimension == 2


def test_geometry_cache_reuses_memory_and_versioned_disk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payoffs = load_game_payoffs("rps")
    digest = payoff_tensor_digest(payoffs)
    calls = 0
    original = geometry_module.analyze_equilibrium_projection_geometry

    def counted(payoff_tensor):
        nonlocal calls
        calls += 1
        return original(payoff_tensor)

    monkeypatch.setattr(
        geometry_module,
        "analyze_equilibrium_projection_geometry",
        counted,
    )
    first_cache = EquilibriumGeometryCache(tmp_path)
    first = first_cache.get(digest, payoffs)
    second = first_cache.get(digest, payoffs)
    reloaded = EquilibriumGeometryCache(tmp_path).get(digest, payoffs)

    assert first is second
    assert calls == 1
    assert reloaded.projection_case == first.projection_case
    assert reloaded.ce.dimension == first.ce.dimension
    assert reloaded.cce.dimension == first.cce.dimension
    assert len(list(tmp_path.glob("v1_*.npz"))) == 1
