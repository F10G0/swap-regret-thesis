"""Lazy dashboard adapter for the experimental comparison workspace."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import logging
from pathlib import Path
import re
import tempfile
from threading import Lock

from experiments.plots import (
    FIGURE_SUFFIXES,
    figure_pair_is_current,
    figure_path,
    publish_figure_pair,
)
from experiments.scenarios.cross_play import replicate_player_seeds
from web.result_groups import result_group_id, result_group_key

from experimental.equilibrium_trajectory.geometry import (
    GEOMETRY_CACHE_VERSION,
    EquilibriumGeometryCache,
)
from experimental.equilibrium_trajectory.projection import (
    COMPARISON_PROJECTION_VERSION,
    TRAJECTORY_RENDER_CACHE_VERSION,
    UNIFIED_COMPARISON_PROJECTION_VERSION,
)
from experimental.equilibrium_trajectory.rendering import (
    TrajectoryComparisonPlotMember,
    plot_result_equilibrium_trajectory_comparison,
)
from experimental.equilibrium_trajectory.settings import (
    DEFAULT_FINAL_INTERVAL_SEGMENTS,
    DEFAULT_TRAJECTORY_COMPARISON_VIEW,
    parse_final_interval_segments,
    parse_trajectory_comparison_view,
)
from experimental.equilibrium_trajectory.web_models import (
    TrajectoryComparisonDefinition,
    TrajectoryComparisonMember,
    TrajectoryComparisonResult,
    comparison_artifact_id,
    comparison_member_colors,
    stable_member_color,
)


logger = logging.getLogger(__name__)


class ExperimentalTrajectoryDashboard:
    """Own all web state and caches for opt-in trajectory rendering."""

    def __init__(self, dashboard_service):
        self.dashboard = dashboard_service
        self.output_dir = (
            dashboard_service.detail_figure_dir
            / "experimental"
            / "equilibrium_trajectory"
        )
        self.geometry_cache = EquilibriumGeometryCache(
            dashboard_service.results_dir
            / "cache"
            / "experimental"
            / "equilibrium_trajectory"
            / "geometry"
        )
        self._publication_lock = Lock()
        self._future_lock = Lock()
        self._generation = 0
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="experimental-equilibrium-trajectory",
        )
        self._futures: dict[
            str,
            Future[TrajectoryComparisonResult],
        ] = {}

    def invalidate(self) -> None:
        with self._future_lock:
            for future in self._futures.values():
                future.cancel()
            self._futures.clear()
        with self._publication_lock:
            self._generation += 1
            if self.output_dir.exists():
                for path in self.output_dir.iterdir():
                    if path.suffix.lower() in FIGURE_SUFFIXES:
                        path.unlink()

    def _member(
        self,
        group_id: str,
        summaries: list[dict],
    ) -> tuple[TrajectoryComparisonMember, tuple, dict]:
        if re.fullmatch(r"[0-9a-f]{16}", group_id) is None:
            raise ValueError("invalid trajectory-comparison member")
        group_rows = [
            summary
            for summary in summaries
            if result_group_id(result_group_key(summary)) == group_id
        ]
        if not group_rows:
            raise KeyError(group_id)
        first = group_rows[0]
        replicate_indices = tuple(sorted({
            int(summary["replicate"])
            for summary in group_rows
        }))
        n_players = int(first["n_players"])
        seed = int(first["seed"])
        seed_schedule = tuple(
            replicate_player_seeds(seed, replicate, n_players)
            for replicate in replicate_indices
        )
        input_paths = tuple(
            self.dashboard._result_group_paths(group_id)
        )
        algorithm_profile = tuple(first["algorithm_profile"])
        profile_label = " vs ".join(
            self.dashboard.algorithm_labels.get(name, name)
            for name in algorithm_profile
        )
        stationary_method = str(first["stationary_method"])
        member = TrajectoryComparisonMember(
            group_id,
            f"{profile_label} · {stationary_method}",
            stable_member_color(group_id),
            algorithm_profile,
            stationary_method,
            replicate_indices,
            seed_schedule,
            input_paths,
        )
        compatibility_key = (
            first["game"],
            first.get("game_payoff_digest", ""),
            first["feedback_mode"],
            first["regret_evaluation"],
            int(first["horizon"]),
            seed,
            len(replicate_indices),
            replicate_indices,
            seed_schedule,
        )
        context = {
            "game": first["game"],
            "game_payoff_digest": first.get(
                "game_payoff_digest", ""
            ),
            "feedback_mode": first["feedback_mode"],
            "regret_evaluation": first["regret_evaluation"],
            "horizon": int(first["horizon"]),
            "seed": seed,
            "replicate_count": len(replicate_indices),
            "replicate_indices": list(replicate_indices),
            "player_seed_schedule": [
                list(seeds) for seeds in seed_schedule
            ],
        }
        return member, compatibility_key, context

    def candidates(self) -> list[dict]:
        summaries = self.dashboard.result_snapshot().summaries
        group_ids = sorted({
            result_group_id(result_group_key(summary))
            for summary in summaries
        })
        candidates = []
        for group_id in group_ids:
            member, compatibility_key, context = self._member(
                group_id,
                summaries,
            )
            candidates.append({
                **member.public_data(),
                **context,
                "compatibility_key": compatibility_key,
            })
        return candidates

    def definition(
        self,
        group_ids: list[str] | tuple[str, ...],
        final_interval_segments: int,
        focus_final_interval: bool,
        comparison_view: str = DEFAULT_TRAJECTORY_COMPARISON_VIEW,
    ) -> TrajectoryComparisonDefinition:
        final_interval_segments = parse_final_interval_segments(
            str(final_interval_segments)
        )
        comparison_view = parse_trajectory_comparison_view(
            comparison_view
        )
        canonical_ids = sorted(set(group_ids))
        if not canonical_ids:
            raise ValueError(
                "select at least one trajectory-comparison member"
            )
        summaries = self.dashboard.result_snapshot().summaries
        resolved = [
            self._member(group_id, summaries)
            for group_id in canonical_ids
        ]
        members = tuple(member for member, _, _ in resolved)
        compatibility_keys = [key for _, key, _ in resolved]
        if any(
            key != compatibility_keys[0]
            for key in compatibility_keys[1:]
        ):
            raise ValueError(
                "trajectory-comparison members must match game, payoff, "
                "feedback, regret evaluation, horizon, seed, replicate "
                "count, replicate indices, and derived player-seed schedule"
            )
        colors = comparison_member_colors(canonical_ids)
        members = tuple(
            replace(member, color=colors[member.group_id])
            for member in members
        )
        artifact_id = comparison_artifact_id(
            compatibility_keys[0],
            members,
            final_interval_segments,
            bool(focus_final_interval),
            GEOMETRY_CACHE_VERSION,
            (
                COMPARISON_PROJECTION_VERSION
                if comparison_view == "geometry"
                else UNIFIED_COMPARISON_PROJECTION_VERSION
            ),
            TRAJECTORY_RENDER_CACHE_VERSION,
            comparison_view,
        )
        return TrajectoryComparisonDefinition(
            members,
            compatibility_keys[0],
            final_interval_segments,
            bool(focus_final_interval),
            artifact_id,
            comparison_view,
        )

    def _output_path(self, artifact_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{24}", artifact_id) is None:
            raise ValueError("invalid trajectory-comparison artifact")
        return self.output_dir / f"trajectory_comparison_{artifact_id}.png"

    def artifact(self, artifact_id: str, figure_format: str = "png") -> Path:
        output_path = figure_path(self._output_path(artifact_id), figure_format)
        if not output_path.is_file():
            raise FileNotFoundError(artifact_id)
        return output_path

    def request(
        self,
        group_ids: list[str] | tuple[str, ...],
        final_interval_segments: int = DEFAULT_FINAL_INTERVAL_SEGMENTS,
        focus_final_interval: bool = False,
        comparison_view: str = DEFAULT_TRAJECTORY_COMPARISON_VIEW,
    ) -> tuple[TrajectoryComparisonResult | None, str | None]:
        definition = self.definition(
            group_ids,
            final_interval_segments,
            focus_final_interval,
            comparison_view,
        )
        output_path = self._output_path(definition.artifact_id)
        input_mtime = max(
            path.stat().st_mtime_ns
            for path in definition.input_paths
        )
        if figure_pair_is_current(output_path, input_mtime):
            return TrajectoryComparisonResult(
                definition,
                output_path,
            ), None

        future_key = f"trajectory-comparison:{definition.artifact_id}"
        scheduled = False
        with self._future_lock:
            future = self._futures.get(future_key)
            if future is None:
                future = self._executor.submit(
                    self._generate,
                    definition,
                    output_path,
                )
                self._futures[future_key] = future
                scheduled = True
        if scheduled or not future.done():
            return None, None
        with self._future_lock:
            self._futures.pop(future_key, None)
        try:
            return future.result(), None
        except Exception as error:
            logger.exception(
                "Trajectory-comparison generation failed for %s",
                definition.artifact_id,
            )
            return None, f"{type(error).__name__}: {error}"

    def _generate(
        self,
        definition: TrajectoryComparisonDefinition,
        output_path: Path,
    ) -> TrajectoryComparisonResult:
        input_state = {
            path: path.stat().st_mtime_ns
            for path in definition.input_paths
        }
        with self._publication_lock:
            generation = self._generation
            self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".trajectory-comparison-",
            dir=self.output_dir,
        ) as temporary_directory:
            temporary_path = (
                Path(temporary_directory) / output_path.name
            )
            plot_result_equilibrium_trajectory_comparison(
                [
                    TrajectoryComparisonPlotMember(
                        member.group_id,
                        member.label,
                        member.color,
                        member.input_paths,
                    )
                    for member in definition.members
                ],
                temporary_path,
                final_interval_segments=(
                    definition.final_interval_segments
                ),
                game_label=self.dashboard.game_presentations[
                    definition.compatibility_key[0]
                ]["label"],
                custom_game_dir=(
                    self.dashboard.game_catalog.custom_game_dir
                ),
                focus_final_interval=(
                    definition.focus_final_interval
                ),
                geometry_cache=self.geometry_cache,
                comparison_view=definition.comparison_view,
            )
            with self._publication_lock:
                if generation != self._generation:
                    raise RuntimeError(
                        "trajectory comparison was invalidated"
                    )
                if any(
                    not path.is_file()
                    or path.stat().st_mtime_ns != mtime
                    for path, mtime in input_state.items()
                ):
                    raise RuntimeError(
                        "comparison members changed during generation"
                    )
                publish_figure_pair(temporary_path, output_path)
        return TrajectoryComparisonResult(definition, output_path)
