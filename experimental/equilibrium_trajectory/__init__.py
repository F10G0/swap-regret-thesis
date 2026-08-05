"""Public facade for experimental equilibrium-trajectory visualization."""

from experimental.equilibrium_trajectory.analysis import (
    EquilibriumTrajectoryComparison,
    EquilibriumTrajectoryProjection,
    analyze_equilibrium_convergence,
    project_equilibrium_trajectory,
    project_equilibrium_trajectory_comparison,
    project_geometry_trajectory,
    project_unified_equilibrium_trajectory_comparison,
)
from experimental.equilibrium_trajectory.geometry import EquilibriumGeometryCache
from experimental.equilibrium_trajectory.rendering import (
    TrajectoryComparisonPlotMember,
    plot_result_equilibrium_trajectory,
    plot_result_equilibrium_trajectory_comparison,
)
from experimental.equilibrium_trajectory.web_models import (
    TrajectoryComparisonDefinition,
    TrajectoryComparisonResult,
)

__all__ = [
    "EquilibriumGeometryCache",
    "EquilibriumTrajectoryComparison",
    "EquilibriumTrajectoryProjection",
    "TrajectoryComparisonPlotMember",
    "TrajectoryComparisonDefinition",
    "TrajectoryComparisonResult",
    "analyze_equilibrium_convergence",
    "plot_result_equilibrium_trajectory",
    "plot_result_equilibrium_trajectory_comparison",
    "project_equilibrium_trajectory",
    "project_equilibrium_trajectory_comparison",
    "project_geometry_trajectory",
    "project_unified_equilibrium_trajectory_comparison",
]
