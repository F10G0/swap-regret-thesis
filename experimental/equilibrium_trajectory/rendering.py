"""Matplotlib rendering for experimental equilibrium trajectories."""

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.plots import save_figure_pair

from config import CUSTOM_GAME_DIR
from experiments.game_catalog import payoff_tensor_digest
from metrics.empirical_distribution import (
    EmpiricalDistributionTrajectory,
    final_interval_checkpoints,
    final_logarithmic_interval_start,
    mean_empirical_distribution_trajectory,
)
from experimental.equilibrium_trajectory.analysis import (
    EquilibriumTrajectoryComparison,
    EquilibriumTrajectoryProjection,
    project_equilibrium_trajectory,
    project_equilibrium_trajectory_comparison,
    project_unified_equilibrium_trajectory_comparison,
)
from experimental.equilibrium_trajectory.geometry import EquilibriumGeometryCache
from experimental.equilibrium_trajectory.projection import (
    DEFAULT_RELATIVE_RENDER_TOLERANCE,
    DEFAULT_SUPPORT_QUERY_CAP,
    ProjectedEquilibriumSet,
)
from experiments.plots.plot_equilibrium_convergence import (
    empirical_distribution_trajectories,
    load_equilibrium_result_inputs,
)


_DEFAULT_GEOMETRY_CACHE = EquilibriumGeometryCache()
_EQUILIBRIUM_ANCHOR_PADDING = 0.06


@dataclass(frozen=True)
class TrajectoryComparisonPlotMember:
    member_id: str
    label: str
    color: str
    input_paths: tuple[str | Path, ...]


def _draw_region(
    axes,
    region: ProjectedEquilibriumSet,
    color: str,
    label: str,
    alpha: float,
    outer: bool = False,
) -> None:
    boundary = region.boundary
    if region.affine_dimension == 0:
        if outer:
            axes.scatter(
                boundary[:, 0],
                boundary[:, 1],
                color=color,
                s=100,
                label=label,
                zorder=3,
            )
        else:
            axes.scatter(
                boundary[:, 0],
                boundary[:, 1],
                facecolors="none",
                edgecolors=color,
                linewidths=2.8,
                s=165,
                label=label,
                clip_on=False,
                zorder=8,
            )
    elif region.affine_dimension == 1:
        axes.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=color,
            linewidth=7 if outer else 3.2,
            linestyle="-" if outer else "--",
            alpha=alpha + 0.2 if outer else 0.95,
            label=label,
            clip_on=outer,
            zorder=2 if outer else 6,
        )
        axes.scatter(
            boundary[:, 0],
            boundary[:, 1],
            color=color if outer else None,
            facecolors=None if outer else "none",
            edgecolors=color,
            linewidths=1.8 if not outer else None,
            s=32 if outer else 48,
            clip_on=outer,
            zorder=3 if outer else 6,
        )
    else:
        closed = np.vstack((boundary, boundary[0]))
        axes.fill(
            closed[:, 0],
            closed[:, 1],
            facecolor=color,
            edgecolor="none" if outer else color,
            hatch=None if outer else "///",
            alpha=alpha,
            label=label,
            zorder=1,
        )
        axes.plot(
            closed[:, 0],
            closed[:, 1],
            color=color,
            linewidth=2.8 if outer else 1.4,
            linestyle="-" if outer else "--",
            zorder=2 if outer else 6,
        )


def _trajectory_view_limits(
    focused_trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_minima = np.min(focused_trajectory, axis=0)
    coordinate_maxima = np.max(focused_trajectory, axis=0)
    coordinate_spans = coordinate_maxima - coordinate_minima
    reference_span = float(np.max(coordinate_spans))
    if reference_span == 0.0:
        reference_span = 1.0
    display_spans = np.where(
        coordinate_spans > 0.0,
        coordinate_spans,
        reference_span,
    )
    centers = (coordinate_minima + coordinate_maxima) / 2.0
    half_ranges = 0.62 * display_spans
    return centers - half_ranges, centers + half_ranges


def _closest_point_on_segment(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    direction = second - first
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        return first.copy()
    position = float(np.dot(point - first, direction) / squared_length)
    return first + np.clip(position, 0.0, 1.0) * direction


def _point_in_polygon(point: np.ndarray, boundary: np.ndarray) -> bool:
    offsets = np.roll(boundary, -1, axis=0) - boundary
    relative = point - boundary
    cross_products = (
        offsets[:, 0] * relative[:, 1]
        - offsets[:, 1] * relative[:, 0]
    )
    tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(boundary))),
    )
    return bool(
        np.all(cross_products >= -tolerance)
        or np.all(cross_products <= tolerance)
    )


def _region_segments(
    region: ProjectedEquilibriumSet,
) -> list[tuple[np.ndarray, np.ndarray]]:
    boundary = np.asarray(region.boundary, dtype=float)
    if region.affine_dimension == 0 or len(boundary) < 2:
        return []
    if region.affine_dimension == 1:
        return [(boundary[0], boundary[-1])]
    return [
        (boundary[position], boundary[(position + 1) % len(boundary)])
        for position in range(len(boundary))
    ]


def _nearest_region_point(
    query_points: np.ndarray,
    region: ProjectedEquilibriumSet,
) -> np.ndarray:
    boundary = np.asarray(region.boundary, dtype=float)
    if region.affine_dimension == 0:
        return boundary[0].copy()
    if region.affine_dimension == 2:
        for point in query_points:
            if _point_in_polygon(point, boundary):
                return point.copy()

    best_point = boundary[0].copy()
    best_distance = float("inf")
    for query in query_points:
        for first, second in _region_segments(region):
            candidate = _closest_point_on_segment(query, first, second)
            distance = float(np.linalg.norm(candidate - query))
            if distance < best_distance:
                best_point = candidate
                best_distance = distance
    return best_point


def _orientation(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> float:
    return float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    tolerance = 1e-12
    if np.any(
        np.maximum(
            np.minimum(first_start, first_end),
            np.minimum(second_start, second_end),
        )
        > np.minimum(
            np.maximum(first_start, first_end),
            np.maximum(second_start, second_end),
        )
        + tolerance
    ):
        return False
    first_pair = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
    )
    second_pair = (
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    return (
        max(first_pair) >= -tolerance
        and min(first_pair) <= tolerance
        and max(second_pair) >= -tolerance
        and min(second_pair) <= tolerance
    )


def _region_visible(
    region: ProjectedEquilibriumSet,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    boundary = np.asarray(region.boundary, dtype=float)
    if np.any(np.all((boundary >= lower) & (boundary <= upper), axis=1)):
        return True
    corners = np.array([
        [lower[0], lower[1]],
        [upper[0], lower[1]],
        [upper[0], upper[1]],
        [lower[0], upper[1]],
    ])
    if (
        region.affine_dimension == 2
        and any(_point_in_polygon(corner, boundary) for corner in corners)
    ):
        return True
    rectangle_segments = [
        (corners[position], corners[(position + 1) % len(corners)])
        for position in range(len(corners))
    ]
    return any(
        _segments_intersect(first, second, box_first, box_second)
        for first, second in _region_segments(region)
        for box_first, box_second in rectangle_segments
    )


def _expand_limits_for_anchor(
    lower: np.ndarray,
    upper: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    padding = _EQUILIBRIUM_ANCHOR_PADDING * (upper - lower)
    return (
        np.minimum(lower, anchor - padding),
        np.maximum(upper, anchor + padding),
    )


def _equilibrium_view_limits(
    focused_trajectory: np.ndarray,
    ce_region: ProjectedEquilibriumSet | None,
    cce_region: ProjectedEquilibriumSet | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _trajectory_view_limits(focused_trajectory)
    if ce_region is not None:
        ce_anchor = _nearest_region_point(
            focused_trajectory,
            ce_region,
        )
        lower, upper = _expand_limits_for_anchor(
            lower,
            upper,
            ce_anchor,
        )
    if (
        cce_region is not None
        and not _region_visible(cce_region, lower, upper)
    ):
        cce_anchor = _nearest_region_point(
            focused_trajectory,
            cce_region,
        )
        lower, upper = _expand_limits_for_anchor(
            lower,
            upper,
            cce_anchor,
        )
    return lower, upper


def _rounded_round_label(round_number: int) -> str:
    magnitude = 10 ** int(math.floor(math.log10(round_number)))
    rounded = int(math.floor(round_number / magnitude + 0.5)) * magnitude
    for scale, suffix in (
        (1_000_000_000, "b"),
        (1_000_000, "m"),
        (1_000, "k"),
    ):
        if rounded >= scale:
            return f"{rounded // scale}{suffix}"
    return str(rounded)


def _informative_horizon_positions(
    horizons: np.ndarray,
) -> list[int]:
    positions = []
    final_position = len(horizons) - 1
    for position, raw_horizon in enumerate(horizons):
        horizon = int(raw_horizon)
        is_power_of_ten = horizon > 0 and 10 ** int(
            math.floor(math.log10(horizon))
        ) == horizon
        if is_power_of_ten or position == final_position:
            positions.append(position)
    return positions


def _plot_equilibrium_trajectory(analysis: EquilibriumTrajectoryProjection, output_path: str | Path, game_name: str,
                                 n_replicates: int, focus_from_checkpoint: int = 0) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(7.2, 5.8))
    view_kind = getattr(analysis, "view_kind", "geometry")
    equilibrium_relative = view_kind == "equilibrium_relative"
    _draw_region(
        axes,
        analysis.cce_region,
        "#60a5fa",
        "CCE (distance = 0)" if equilibrium_relative else "Projected CCE",
        0.2,
        outer=True,
    )
    _draw_region(
        axes,
        analysis.ce_region,
        "#f59e0b",
        "CE (origin)" if equilibrium_relative else "Projected CE",
        0.3,
    )

    trajectory = analysis.projected_trajectory
    focus_from_checkpoint = min(
        focus_from_checkpoint,
        len(trajectory) - 1,
    )
    rendered_from_checkpoint = max(0, focus_from_checkpoint - 1)
    rendered_trajectory = trajectory[rendered_from_checkpoint:]
    trajectory_label = "Empirical trajectory" if n_replicates == 1 else "Mean empirical trajectory"
    axes.plot(
        rendered_trajectory[:, 0],
        rendered_trajectory[:, 1],
        color="#4b5563",
        marker="o",
        markerfacecolor="#6b7280",
        markeredgecolor="#ffffff",
        markersize=4.5,
        linewidth=1.9,
        label=trajectory_label,
        zorder=4,
    )
    start_position = focus_from_checkpoint
    if start_position < len(trajectory) - 1:
        axes.scatter(
            trajectory[start_position, 0],
            trajectory[start_position, 1],
            color="#16a34a",
            edgecolors="#ffffff",
            linewidths=1.8,
            marker="o",
            s=115,
            label="Start",
            zorder=7,
        )
        axes.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color="#dc2626",
            edgecolors="#ffffff",
            linewidths=1.8,
            marker="X",
            s=125,
            label="End",
            zorder=7,
        )
    else:
        axes.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color="#7c3aed",
            edgecolors="#ffffff",
            linewidths=1.8,
            marker="D",
            s=120,
            label="Start and end",
            zorder=7,
        )

    focused_trajectory = trajectory[focus_from_checkpoint:]
    lower_limits, upper_limits = _equilibrium_view_limits(
        focused_trajectory,
        analysis.ce_region,
        analysis.cce_region,
    )
    if equilibrium_relative:
        lower_limits[1] = 0.0
    axes.set_xlim(lower_limits[0], upper_limits[0])
    axes.set_ylim(lower_limits[1], upper_limits[1])

    label_positions = _informative_horizon_positions(
        analysis.empirical.horizons
    )
    for position in label_positions:
        if position < focus_from_checkpoint:
            continue
        point = trajectory[position]
        horizon = int(analysis.empirical.horizons[position])
        label = _rounded_round_label(horizon)
        offset = (8, -17) if position == len(trajectory) - 1 else (8, 8)
        axes.annotate(
            label,
            point,
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
            fontweight="semibold",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
            zorder=8,
        )

    axis_labels = getattr(
        analysis,
        "axis_labels",
        ("Projected component 1", "Projected component 2"),
    )
    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.grid(alpha=0.2)
    axes.legend()
    if equilibrium_relative:
        title = (
            "Equilibrium-Relative Trajectory"
            if n_replicates == 1
            else f"Equilibrium-Relative Mean Trajectory ({n_replicates} replicates)"
        )
    else:
        title = "Projected Joint-Distribution Trajectory" if n_replicates == 1 else f"Projected Mean Joint-Distribution Trajectory ({n_replicates} replicates)"
    axes.set_title(f"{game_name}: {title}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pair(figure, output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_equilibrium_trajectory_comparison(
    analysis: EquilibriumTrajectoryComparison,
    plot_members: list[TrajectoryComparisonPlotMember],
    output_path: str | Path,
    game_name: str,
    focus_from_checkpoint: int,
) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(7.8, 5.8))
    equilibrium_relative = analysis.view_kind in {
        "equilibrium_relative",
        "unified_equilibrium_relative",
    }
    unified = analysis.view_kind == "unified_equilibrium_relative"
    _draw_region(
        axes,
        analysis.cce_region,
        "#60a5fa",
        (
            "CCE projection (distance = 0)"
            if unified
            else "CCE (distance = 0)"
            if equilibrium_relative
            else "Projected CCE"
        ),
        0.2,
        outer=True,
    )
    _draw_region(
        axes,
        analysis.ce_region,
        "#f59e0b",
        (
            "CE projection (membership not implied)"
            if unified
            else "CE (origin)"
            if equilibrium_relative
            else "Projected CE"
        ),
        0.3,
    )

    focused_coordinates = []
    for member_position, (member, projected) in enumerate(
        zip(plot_members, analysis.members)
    ):
        trajectory = projected.projected_trajectory
        focus_position = min(
            focus_from_checkpoint,
            len(trajectory) - 1,
        )
        rendered_from = max(0, focus_position - 1)
        rendered = trajectory[rendered_from:]
        focused_coordinates.append(trajectory[focus_position:])
        axes.plot(
            rendered[:, 0],
            rendered[:, 1],
            color=member.color,
            marker="o",
            markerfacecolor=member.color,
            markeredgecolor="#ffffff",
            markersize=4.5,
            linewidth=1.9,
            zorder=4,
        )
        if focus_position < len(trajectory) - 1:
            axes.scatter(
                trajectory[focus_position, 0],
                trajectory[focus_position, 1],
                color=member.color,
                edgecolors="#ffffff",
                linewidths=1.8,
                marker="o",
                s=105,
                label=(
                    "Focused start"
                    if member_position == 0
                    else None
                ),
                zorder=7,
            )
            axes.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=member.color,
                edgecolors="#ffffff",
                linewidths=1.8,
                marker="X",
                s=115,
                label="End" if member_position == 0 else None,
                zorder=7,
            )
        else:
            axes.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=member.color,
                edgecolors="#ffffff",
                linewidths=1.8,
                marker="D",
                s=110,
                label=(
                    "Start / end"
                    if member_position == 0
                    else None
                ),
                zorder=7,
            )

    focused = np.vstack(focused_coordinates)
    lower_limits, upper_limits = _equilibrium_view_limits(
        focused,
        analysis.ce_region,
        analysis.cce_region,
    )
    if equilibrium_relative:
        lower_limits[1] = 0.0
    axes.set_xlim(lower_limits[0], upper_limits[0])
    axes.set_ylim(lower_limits[1], upper_limits[1])
    axes.set_xlabel(analysis.axis_labels[0])
    axes.set_ylabel(analysis.axis_labels[1])
    axes.grid(alpha=0.2)
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels)
    title = (
        "Unified Equilibrium-Relative Comparison"
        if unified
        else "Equilibrium-Relative Trajectory Comparison"
        if equilibrium_relative
        else "Equilibrium Geometry Trajectory Comparison"
    )
    axes.set_title(f"{game_name}: {title}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pair(figure, output_path, bbox_inches="tight")
    plt.close(figure)


def _trajectory_checkpoints_and_focus(
    horizon: int,
    final_interval_segments: int,
    focus_final_interval: bool,
) -> tuple[np.ndarray, int]:
    checkpoints = final_interval_checkpoints(
        horizon,
        final_interval_segments,
    )
    if not focus_final_interval or horizon == 1:
        return checkpoints, 0
    interval_start = final_logarithmic_interval_start(horizon)
    interval_position = int(np.searchsorted(
        checkpoints,
        interval_start,
    ))
    represented_from = max(0, interval_position - 1)
    return (
        checkpoints[represented_from:],
        interval_position - represented_from,
    )


def _focused_empirical(
    empirical: EmpiricalDistributionTrajectory,
    focus_from_checkpoint: int,
) -> EmpiricalDistributionTrajectory:
    if not focus_from_checkpoint:
        return empirical
    return EmpiricalDistributionTrajectory(
        empirical.action_shape,
        empirical.horizons[focus_from_checkpoint:],
        empirical.vectors[focus_from_checkpoint:],
    )


def _equilibrium_geometry(
    payoff_tensor: np.ndarray,
    geometry_cache: EquilibriumGeometryCache | None,
):
    cache = geometry_cache or _DEFAULT_GEOMETRY_CACHE
    return cache.get(
        payoff_tensor_digest(payoff_tensor),
        payoff_tensor,
    )


def plot_result_equilibrium_trajectory_comparison(
    members: Iterable[TrajectoryComparisonPlotMember],
    output_path: str | Path,
    final_interval_segments: int = 10,
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    relative_render_tolerance: float | None = DEFAULT_RELATIVE_RENDER_TOLERANCE,
    game_label: str | None = None,
    custom_game_dir: str | Path = CUSTOM_GAME_DIR,
    focus_final_interval: bool = False,
    geometry_cache: EquilibriumGeometryCache | None = None,
    comparison_view: str = "geometry",
) -> EquilibriumTrajectoryComparison:
    ordered_members = sorted(members, key=lambda member: member.member_id)
    if not ordered_members:
        raise ValueError("at least one trajectory-comparison member is required")
    if len({member.member_id for member in ordered_members}) != len(ordered_members):
        raise ValueError("trajectory-comparison member ids must be unique")

    loaded = [
        load_equilibrium_result_inputs(
            member.input_paths,
            custom_game_dir,
        )
        for member in ordered_members
    ]
    game_names = {game_name for game_name, _, _ in loaded}
    if len(game_names) != 1:
        raise ValueError("trajectory-comparison members must use the same game")
    game_name = game_names.pop()
    payoff_tensor = loaded[0][1]
    if any(
        not np.array_equal(member_payoffs, payoff_tensor)
        for _, member_payoffs, _ in loaded[1:]
    ):
        raise ValueError("trajectory-comparison members use different payoff tensors")
    horizons = {
        len(profile)
        for _, _, profiles in loaded
        for profile in profiles
    }
    if len(horizons) != 1:
        raise ValueError("trajectory-comparison members must use the same horizon")
    horizon = horizons.pop()
    checkpoints, focus_from_checkpoint = _trajectory_checkpoints_and_focus(
        horizon,
        final_interval_segments,
        focus_final_interval,
    )

    empiricals = []
    for _, _, profiles in loaded:
        replicate_trajectories = empirical_distribution_trajectories(
            profiles,
            payoff_tensor.shape[1:],
            checkpoints,
        )
        empiricals.append(
            mean_empirical_distribution_trajectory(
                replicate_trajectories
            )
        )
    fitting_empiricals = [
        _focused_empirical(empirical, focus_from_checkpoint)
        for empirical in empiricals
    ]
    member_ids = [member.member_id for member in ordered_members]
    if comparison_view == "geometry":
        analysis = project_equilibrium_trajectory_comparison(
            payoff_tensor,
            empiricals,
            member_ids=member_ids,
            support_query_cap=support_query_cap,
            geometry=_equilibrium_geometry(
                payoff_tensor,
                geometry_cache,
            ),
            relative_render_tolerance=relative_render_tolerance,
            fit_empiricals=fitting_empiricals,
        )
    elif comparison_view == "unified":
        analysis = project_unified_equilibrium_trajectory_comparison(
            payoff_tensor,
            empiricals,
            member_ids=member_ids,
            fit_empiricals=fitting_empiricals,
        )
    else:
        raise ValueError(
            "comparison_view must be 'geometry' or 'unified'"
        )
    _plot_equilibrium_trajectory_comparison(
        analysis,
        ordered_members,
        output_path,
        game_label or game_name,
        focus_from_checkpoint,
    )
    return analysis


def plot_result_equilibrium_trajectory(input_paths: str | Path | Iterable[str | Path], output_path: str | Path,
                                       final_interval_segments: int = 10,
                                       support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
                                       relative_render_tolerance: float | None = DEFAULT_RELATIVE_RENDER_TOLERANCE,
                                       game_label: str | None = None,
                                       custom_game_dir: str | Path = CUSTOM_GAME_DIR,
                                       focus_final_interval: bool = False,
                                       geometry_cache: EquilibriumGeometryCache | None = None) -> None:
    game_name, payoff_tensor, profiles = load_equilibrium_result_inputs(
        input_paths,
        custom_game_dir,
    )
    checkpoints, focus_from_checkpoint = _trajectory_checkpoints_and_focus(
        len(profiles[0]),
        final_interval_segments,
        focus_final_interval,
    )
    empirical = mean_empirical_distribution_trajectory(
        empirical_distribution_trajectories(
            profiles,
            payoff_tensor.shape[1:],
            checkpoints,
        )
    )
    analysis = project_equilibrium_trajectory(
        payoff_tensor,
        empirical,
        support_query_cap,
        _equilibrium_geometry(payoff_tensor, geometry_cache),
        relative_render_tolerance,
        fit_empirical=_focused_empirical(
            empirical,
            focus_from_checkpoint,
        ),
    )
    _plot_equilibrium_trajectory(
        analysis,
        output_path,
        game_label or game_name,
        len(profiles),
        focus_from_checkpoint,
    )
