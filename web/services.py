from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

import numpy as np

from config import BANDIT_REPLICATES, CUSTOM_GAME_DIR, HORIZON, SEED
from experiments.algorithm_labels import algorithm_label, algorithm_profile_label
from experiments.game_catalog import GameCatalog, GameDefinition
from experiments.games import PAYOFF_FACTORIES
from experiments.results import iter_result_rows
from experiments.spec import ExperimentSpec
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.experiment_modes import FEEDBACK_MODES, FeedbackMode
from web.jobs import Job, JobContext, JobManager
from web.presentations import GAME_PRESENTATIONS
from web.result_groups import result_group_filenames
from web.result_index import ResultIndex, ResultSnapshot
from web.validation import DEFAULT_TRAJECTORY_POINTS, ExperimentForm, validate_leaf_filename


logger = logging.getLogger(__name__)


class PlotUpdateError(RuntimeError):
    pass


class DashboardService:
    def __init__(
        self,
        results_dir: str | Path,
        raw_dir: str | Path,
        figure_dir: str | Path,
        job_manager: JobManager | None = None,
        custom_game_dir: str | Path = CUSTOM_GAME_DIR,
    ):
        self.results_dir = Path(results_dir)
        self.raw_dir = Path(raw_dir)
        self.figure_dir = Path(figure_dir)
        self.game_catalog = GameCatalog(custom_game_dir)
        self.jobs = job_manager or JobManager()
        self.result_index = ResultIndex(self.raw_dir)
        self._detail_figure_lock = Lock()
        self._detail_figure_generation = 0
        self._convergence_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="equilibrium-convergence",
        )
        self._convergence_future_lock = Lock()
        self._convergence_futures: dict[str, Future[dict[str, Path]]] = {}

    @property
    def games(self) -> list[str]:
        return list(self.game_definitions)

    @property
    def game_definitions(self) -> dict[str, GameDefinition]:
        return self.game_catalog.definitions()

    @property
    def game_player_counts(self) -> dict[str, int]:
        return {game_id: definition.n_players for game_id, definition in self.game_definitions.items()}

    @property
    def game_presentations(self) -> dict[str, dict[str, str]]:
        presentations = {}
        for game_name, definition in self.game_definitions.items():
            configured = GAME_PRESENTATIONS.get(game_name)
            presentations[game_name] = (
                dict(configured)
                if configured is not None
                else {
                    "label": definition.label,
                    "description": definition.description,
                }
            )
        return presentations

    def supports_matrix_figures(self, game_name: str) -> bool:
        return game_name in PAYOFF_FACTORIES

    def supports_equilibrium_distance(self, game_name: str) -> bool:
        return game_name in self.game_definitions

    def supports_equilibrium_trajectory(self, game_name: str) -> bool:
        return game_name in self.game_definitions

    def custom_games(self) -> tuple[list[GameDefinition], list[str]]:
        return self.game_catalog.custom_definitions()

    def create_custom_game(self, name: str, n_players, action_counts, seed) -> GameDefinition:
        return self.game_catalog.create_random(name, n_players, action_counts, seed)

    def delete_custom_game(self, game_id: str) -> GameDefinition:
        def operation() -> GameDefinition:
            result_prefix = f"{game_id}_"
            if self.raw_dir.exists() and any(path.name.startswith(result_prefix) for path in self.raw_dir.glob("*.csv")):
                raise ValueError("delete the recorded experiments for this game before deleting the game")
            definition = self.game_catalog.delete(game_id)
            self._clear_figure_files(game_id)
            return definition

        return self.jobs.run_maintenance(operation)

    def custom_game_inspection(self, game_id: str) -> dict:
        definition = self.game_definitions.get(game_id)
        if definition is None or definition.source != "custom":
            raise KeyError(game_id)
        payoff_tensor = self.game_catalog.load(game_id)
        return {
            "definition": definition.public_data(),
            "shape": payoff_tensor.shape,
            "minimum": float(np.min(payoff_tensor)),
            "maximum": float(np.max(payoff_tensor)),
            "mean": float(np.mean(payoff_tensor)),
        }

    def custom_game_payoff_slice(self, game_id: str, payoff_player: int, row_player: int, column_player: int, fixed_actions: list[int]) -> dict:
        definition = self.game_definitions.get(game_id)
        if definition is None or definition.source != "custom":
            raise KeyError(game_id)
        if not 0 <= payoff_player < definition.n_players:
            raise ValueError("invalid payoff player")
        if not 0 <= row_player < definition.n_players or not 0 <= column_player < definition.n_players:
            raise ValueError("invalid axis player")
        if row_player == column_player:
            raise ValueError("row and column players must be different")
        if len(fixed_actions) != definition.n_players:
            raise ValueError("provide one fixed action per player")
        for player, action in enumerate(fixed_actions):
            if not 0 <= action < definition.action_counts[player]:
                raise ValueError(f"invalid fixed action for player {player}")

        payoff_tensor = self.game_catalog.load(game_id)
        values = np.empty((definition.action_counts[row_player], definition.action_counts[column_player]))
        joint_action = list(fixed_actions)
        for row_action in range(values.shape[0]):
            joint_action[row_player] = row_action
            for column_action in range(values.shape[1]):
                joint_action[column_player] = column_action
                values[row_action, column_action] = payoff_tensor[(payoff_player, *joint_action)]
        return {
            "game": game_id,
            "payoff_player": payoff_player,
            "row_player": row_player,
            "column_player": column_player,
            "fixed_actions": fixed_actions,
            "values": values.tolist(),
        }

    def custom_game_file(self, game_id: str) -> Path:
        definition = self.game_definitions.get(game_id)
        if definition is None or definition.source != "custom":
            raise KeyError(game_id)
        return self.game_catalog.custom_path(game_id)

    @property
    def feedback_modes(self) -> dict[str, dict]:
        return {
            name: {
                "label": mode.label,
                "algorithms": list(mode.algorithms),
            }
            for name, mode in FEEDBACK_MODES.items()
        }

    @property
    def algorithms_by_feedback_mode(self) -> dict[str, list[str]]:
        return {
            name: list(mode.algorithms)
            for name, mode in FEEDBACK_MODES.items()
        }

    @property
    def algorithm_labels(self) -> dict[str, str]:
        return {name: algorithm_label(name) for algorithms in self.algorithms_by_feedback_mode.values() for name in algorithms}

    def default_form_state(self) -> dict:
        feedback_mode = "full_information"
        first_algorithm = self.algorithms_by_feedback_mode[feedback_mode][0]
        game = self.games[0]
        return {
            "game": game,
            "feedback_mode": feedback_mode,
            "regret_evaluation": "expected",
            "algorithm_names": [first_algorithm] * self.game_player_counts[game],
            "horizon": HORIZON,
            "seed": SEED,
            "replicate": 0,
            "replicates": BANDIT_REPLICATES,
        }

    def _spec(self, form: ExperimentForm, algorithm_names: list[str] | None = None, replicate: int | None = None) -> ExperimentSpec:
        names = algorithm_names or form.algorithm_names
        return ExperimentSpec(
            game_name=form.game,
            feedback_mode=form.feedback_mode,
            algorithm_names=tuple(names),
            horizon=form.horizon,
            seed=form.seed,
            replicate=form.replicate if replicate is None else replicate,
            regret_evaluation=form.regret_evaluation,
        )

    def _replicates(self, form: ExperimentForm) -> range:
        return range(form.replicate, form.replicate + form.replicates)

    def _missing_runs(self, form: ExperimentForm, profiles: list[list[str]]) -> tuple[list[tuple[list[str], int]], int]:
        requested = [(profile, replicate) for profile in profiles for replicate in self._replicates(form)]
        reserved = self.jobs.reserved_resources()
        missing = []
        for profile, replicate in requested:
            run_id = self._spec(form, profile, replicate).run_id
            if run_id not in reserved and not (self.raw_dir / f"{run_id}.csv").exists():
                missing.append((profile, replicate))
        return missing, len(requested) - len(missing)

    def _run_experiments(self, form: ExperimentForm, mode: FeedbackMode, missing_runs: list[tuple[list[str], int]], skipped_count: int, job: JobContext) -> str:
        for algorithm_names, replicate in missing_runs:
            job.check_cancelled()
            mode.runner(
                game_name=form.game, algorithm_names=algorithm_names, horizon=form.horizon, seed=form.seed, replicate=replicate,
                output_dir=self.raw_dir, should_cancel=lambda: job.cancelled, custom_game_dir=self.game_catalog.custom_game_dir,
                regret_evaluation=form.regret_evaluation,
            )
            job.advance()
        job.check_cancelled()
        try:
            self._publish_plots(form.game)
        except Exception as error:
            raise PlotUpdateError(f"experiments were saved, but their figures could not be rebuilt: {error}") from error
        return f"Completed {len(missing_runs)} run(s); skipped {skipped_count} existing or queued"

    def submit_experiment(self, form: ExperimentForm) -> Job:
        mode = FEEDBACK_MODES[form.feedback_mode]
        missing_runs, skipped_count = self._missing_runs(form, [form.algorithm_names])
        if not missing_runs:
            raise FileExistsError("all requested replicates already exist or are queued")

        def operation(job: JobContext) -> str:
            return self._run_experiments(form, mode, missing_runs, skipped_count, job)

        return self.jobs.submit(
            f"{form.game}: {algorithm_profile_label(form.algorithm_names)}",
            operation,
            total=len(missing_runs),
            resource_keys={self._spec(form, names, replicate).run_id for names, replicate in missing_runs},
        )

    def submit_all_pairs(self, form: ExperimentForm) -> tuple[Job, int, int]:
        definition = self.game_definitions[form.game]
        if definition.source != "builtin" or definition.n_players != 2:
            raise ValueError("all-pairs runs are available only for built-in two-player games")
        mode = FEEDBACK_MODES[form.feedback_mode]
        pairs = [
            [algorithm_player_0, algorithm_player_1]
            for algorithm_player_0 in mode.algorithms
            for algorithm_player_1 in mode.algorithms
        ]
        missing_runs, skipped_count = self._missing_runs(form, pairs)
        if not missing_runs:
            raise FileExistsError("all requested runs already exist or are queued")

        def operation(job: JobContext) -> str:
            return self._run_experiments(form, mode, missing_runs, skipped_count, job)

        job = self.jobs.submit(
            f"{form.game}: all {form.feedback_mode} pairs",
            operation,
            total=len(missing_runs),
            resource_keys={self._spec(form, names, replicate).run_id for names, replicate in missing_runs},
        )
        return job, len(missing_runs), skipped_count

    def submit_plot_rebuild(self) -> Job:
        return self.jobs.submit(
            "Rebuild all figures",
            lambda job: self._rebuild_all_plots(job),
            reload_page=False,
        )

    def _rebuild_all_plots(self, job: JobContext) -> str:
        job.check_cancelled()
        self._publish_plots()
        job.advance()
        return "Rebuilt all figures"

    def _publish_plots(self, game_name: str | None = None) -> None:
        from experiments.plots.plot_regret import (
            plot_all_results,
            plot_selected_results,
        )

        self.figure_dir.parent.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".figures-",
            dir=self.figure_dir.parent,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            if game_name is None:
                plot_all_results(self.raw_dir, temporary_path, skip_invalid=True)
                matches_target = lambda path: path.suffix.lower() == ".png"
            else:
                plot_selected_results(game_name, self.raw_dir, temporary_path, skip_invalid=True)
                matches_target = lambda path: (
                    path.suffix.lower() == ".png"
                    and path.name.startswith(f"{game_name}_")
                )

            generated_names = {
                path.name
                for path in temporary_path.glob("*.png")
            }
            for generated_path in temporary_path.glob("*.png"):
                os.replace(generated_path, self.figure_dir / generated_path.name)

            for old_path in self.figure_dir.iterdir():
                if (
                    old_path.is_file()
                    and matches_target(old_path)
                    and old_path.name not in generated_names
                ):
                    old_path.unlink()

    def _clear_figure_files(self, game_name: str | None = None) -> None:
        if not self.figure_dir.exists():
            return
        for path in self.figure_dir.glob("*.png"):
            if game_name is None or path.name.startswith(f"{game_name}_"):
                path.unlink()

    @property
    def detail_figure_dir(self) -> Path:
        return self.figure_dir / "details"

    def _result_group_paths(self, group_id: str) -> list[Path]:
        if re.fullmatch(r"[0-9a-f]{16}", group_id) is None:
            raise ValueError("invalid result group")
        filenames = result_group_filenames(self.result_snapshot().summaries, group_id)
        paths = [self.raw_dir / validate_leaf_filename(filename, ".csv") for filename in filenames]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(group_id)
        return paths

    def _group_cache_stem(self, group_id: str, input_paths: list[Path]) -> str:
        membership = "\n".join(path.name for path in input_paths)
        return f"{group_id}_{sha256(membership.encode('utf-8')).hexdigest()[:8]}"

    def _equilibrium_figure_path(self, game_name: str, equilibrium: str) -> Path:
        if not self.supports_matrix_figures(game_name):
            raise ValueError(f"unknown game: {game_name}")
        if equilibrium not in {"ce", "cce"}:
            raise ValueError(f"unknown equilibrium concept: {equilibrium}")
        return PRECOMPUTED_EQUILIBRIUM_DIR / equilibrium_figure_filename(game_name, equilibrium)

    def precomputed_equilibrium_figure(self, game_name: str, equilibrium: str) -> Path:
        output_path = self._equilibrium_figure_path(game_name, equilibrium)
        if not output_path.is_file():
            raise FileNotFoundError(f"missing precomputed equilibrium figure: {output_path.name}")
        return output_path

    def joint_action_figure(self, filename: str) -> Path:
        filename = validate_leaf_filename(filename, ".csv")
        input_path = self.raw_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(filename)
        return self._joint_action_figure([input_path], input_path.stem)

    def group_joint_action_figure(self, group_id: str) -> Path:
        input_paths = self._result_group_paths(group_id)
        cache_stem = self._group_cache_stem(group_id, input_paths)
        return self._joint_action_figure(input_paths, f"{cache_stem}_replicate_mean")

    def _joint_action_figure(self, input_paths: list[Path], cache_stem: str) -> Path:
        game_name = next(iter_result_rows(input_paths[0]))["game"]
        if not self.supports_matrix_figures(game_name):
            raise ValueError(f"joint-action heatmaps are unavailable for {game_name}")

        output_path = self.detail_figure_dir / f"{cache_stem}_joint_actions_blue_lower_origin.png"
        input_mtime = max(path.stat().st_mtime_ns for path in input_paths)
        with self._detail_figure_lock:
            if output_path.is_file() and output_path.stat().st_mtime_ns >= input_mtime:
                return output_path

            from experiments.plots.plot_joint_actions import plot_joint_actions

            self.detail_figure_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".joint-actions-", dir=self.detail_figure_dir) as temporary_directory:
                temporary_path = Path(temporary_directory) / output_path.name
                plot_joint_actions(input_paths, temporary_path)
                os.replace(temporary_path, output_path)
        return output_path

    def _convergence_figure_paths(
        self,
        filename: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> tuple[Path, dict[str, Path]]:
        filename = validate_leaf_filename(filename, ".csv")
        input_path = self.raw_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(filename)
        game_name = next(iter_result_rows(input_path))["game"]
        if not self.supports_equilibrium_distance(game_name):
            raise ValueError(f"equilibrium distance is unavailable for {game_name}")
        paths = {
            "distance": self.detail_figure_dir / f"{input_path.stem}_equilibrium_distance.png",
        }
        if self.supports_equilibrium_trajectory(game_name):
            first_node = "hide_round_1" if hide_first else "from_round_1"
            paths["trajectory"] = self.detail_figure_dir / f"{input_path.stem}_p{trajectory_points}_{first_node}_equilibrium_trajectory.png"
        return input_path, paths

    def _group_convergence_figure_paths(
        self,
        group_id: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> tuple[list[Path], dict[str, Path], str]:
        input_paths = self._result_group_paths(group_id)
        game_name = next(iter_result_rows(input_paths[0]))["game"]
        if not self.supports_equilibrium_distance(game_name):
            raise ValueError(f"equilibrium distance is unavailable for {game_name}")
        cache_stem = self._group_cache_stem(group_id, input_paths)
        paths = {
            "distance": self.detail_figure_dir / f"{cache_stem}_replicate_mean_equilibrium_distance.png",
        }
        if self.supports_equilibrium_trajectory(game_name):
            first_node = "hide_round_1" if hide_first else "from_round_1"
            paths["trajectory"] = self.detail_figure_dir / f"{cache_stem}_p{trajectory_points}_{first_node}_replicate_mean_equilibrium_trajectory.png"
        return input_paths, paths, cache_stem

    def _current_convergence_path(self, filename: str, figure: str, trajectory_points: int, hide_first: bool) -> Path | None:
        input_path, paths = self._convergence_figure_paths(filename, trajectory_points, hide_first)
        if figure not in paths:
            raise ValueError(f"unknown equilibrium convergence figure: {figure}")
        path = paths[figure]
        if path.is_file() and path.stat().st_mtime_ns >= input_path.stat().st_mtime_ns:
            return path
        return None

    def request_equilibrium_convergence_figure(
        self,
        filename: str,
        figure: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> tuple[Path | None, str | None]:
        future_key = f"{filename}:{figure}" if figure == "distance" else f"{filename}:trajectory:p{trajectory_points}:hide{int(hide_first)}"
        current_path = self._current_convergence_path(filename, figure, trajectory_points, hide_first)
        if current_path is not None:
            with self._convergence_future_lock:
                future = self._convergence_futures.get(future_key)
                if future is not None and future.done():
                    self._convergence_futures.pop(future_key, None)
            return current_path, None

        scheduled = False
        with self._convergence_future_lock:
            future = self._convergence_futures.get(future_key)
            if future is None:
                future = self._convergence_executor.submit(
                    self._generate_equilibrium_convergence_figure,
                    filename,
                    figure,
                    trajectory_points,
                    hide_first,
                )
                self._convergence_futures[future_key] = future
                scheduled = True
        if scheduled or not future.done():
            return None, None

        with self._convergence_future_lock:
            self._convergence_futures.pop(future_key, None)
        try:
            return future.result()[figure], None
        except Exception as error:
            logger.exception("Equilibrium convergence figure generation failed for %s", filename)
            return None, f"{type(error).__name__}: {error}"

    def request_group_equilibrium_convergence_figure(
        self,
        group_id: str,
        figure: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> tuple[Path | None, str | None]:
        input_paths, paths, cache_stem = self._group_convergence_figure_paths(group_id, trajectory_points, hide_first)
        if figure not in paths:
            raise ValueError(f"unknown equilibrium convergence figure: {figure}")
        input_mtime = max(path.stat().st_mtime_ns for path in input_paths)
        path = paths[figure]
        if path.is_file() and path.stat().st_mtime_ns >= input_mtime:
            return path, None

        future_key = f"group:{cache_stem}:{figure}" if figure == "distance" else f"group:{cache_stem}:trajectory:p{trajectory_points}:hide{int(hide_first)}"
        scheduled = False
        with self._convergence_future_lock:
            future = self._convergence_futures.get(future_key)
            if future is None:
                future = self._convergence_executor.submit(
                    self._generate_group_equilibrium_convergence_figure,
                    group_id,
                    figure,
                    trajectory_points,
                    hide_first,
                )
                self._convergence_futures[future_key] = future
                scheduled = True
        if scheduled or not future.done():
            return None, None

        with self._convergence_future_lock:
            self._convergence_futures.pop(future_key, None)
        try:
            return future.result()[figure], None
        except Exception as error:
            logger.exception("Equilibrium convergence figure generation failed for group %s", group_id)
            return None, f"{type(error).__name__}: {error}"

    def equilibrium_convergence_figures(
        self,
        filename: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> dict[str, Path]:
        input_path, output_paths = self._convergence_figure_paths(filename, trajectory_points, hide_first)
        return self._generate_equilibrium_convergence_figures([input_path], output_paths, trajectory_points, hide_first)

    def _generate_equilibrium_convergence_figure(
        self,
        filename: str,
        figure: str,
        trajectory_points: int,
        hide_first: bool,
    ) -> dict[str, Path]:
        input_path, output_paths = self._convergence_figure_paths(filename, trajectory_points, hide_first)
        return self._generate_equilibrium_convergence_figures(
            [input_path],
            output_paths,
            trajectory_points,
            hide_first,
            figure,
        )

    def group_equilibrium_convergence_figures(
        self,
        group_id: str,
        trajectory_points: int = DEFAULT_TRAJECTORY_POINTS,
        hide_first: bool = False,
    ) -> dict[str, Path]:
        input_paths, output_paths, _ = self._group_convergence_figure_paths(group_id, trajectory_points, hide_first)
        return self._generate_equilibrium_convergence_figures(input_paths, output_paths, trajectory_points, hide_first)

    def _generate_group_equilibrium_convergence_figure(
        self,
        group_id: str,
        figure: str,
        trajectory_points: int,
        hide_first: bool,
    ) -> dict[str, Path]:
        input_paths, output_paths, _ = self._group_convergence_figure_paths(group_id, trajectory_points, hide_first)
        return self._generate_equilibrium_convergence_figures(
            input_paths,
            output_paths,
            trajectory_points,
            hide_first,
            figure,
        )

    def _generate_equilibrium_convergence_figures(
        self,
        input_paths: list[Path],
        output_paths: dict[str, Path],
        trajectory_points: int,
        hide_first: bool,
        requested_figure: str | None = None,
    ) -> dict[str, Path]:
        with self._detail_figure_lock:
            input_state = {path: path.stat().st_mtime_ns for path in input_paths}
            input_mtime = max(input_state.values())
            needed = {
                name
                for name, path in output_paths.items()
                if not path.is_file() or path.stat().st_mtime_ns < input_mtime
            }
            if requested_figure is not None:
                if requested_figure not in output_paths:
                    raise ValueError(f"unknown equilibrium convergence figure: {requested_figure}")
                needed &= {requested_figure}
            if not needed:
                return output_paths
            generation = self._detail_figure_generation
            self.detail_figure_dir.mkdir(parents=True, exist_ok=True)

        from experiments.plots.plot_equilibrium_convergence import (
            plot_result_equilibrium_convergence,
            plot_result_equilibrium_distance,
            plot_result_equilibrium_trajectory,
        )

        with tempfile.TemporaryDirectory(prefix=".equilibrium-convergence-", dir=self.detail_figure_dir) as temporary_directory:
            temporary_paths = {name: Path(temporary_directory) / path.name for name, path in output_paths.items()}
            game_label = self.game_presentations[next(iter_result_rows(input_paths[0]))["game"]]["label"]

            def publish(name: str) -> None:
                with self._detail_figure_lock:
                    if generation != self._detail_figure_generation:
                        raise RuntimeError("equilibrium convergence figure generation was invalidated")
                    if any(not path.is_file() or path.stat().st_mtime_ns != mtime for path, mtime in input_state.items()):
                        raise RuntimeError("experiment group changed while equilibrium convergence figures were generated")
                    os.replace(temporary_paths[name], output_paths[name])

            if needed == {"distance", "trajectory"}:
                plot_result_equilibrium_convergence(
                    input_paths,
                    temporary_paths["distance"],
                    temporary_paths["trajectory"],
                    trajectory_points=trajectory_points,
                    game_label=game_label,
                    distance_ready=lambda: publish("distance"),
                    custom_game_dir=self.game_catalog.custom_game_dir,
                    hide_first=hide_first,
                )
            elif needed == {"distance"}:
                plot_result_equilibrium_distance(
                    input_paths, temporary_paths["distance"], game_label=game_label,
                    custom_game_dir=self.game_catalog.custom_game_dir,
                )
            else:
                plot_result_equilibrium_trajectory(
                    input_paths,
                    temporary_paths["trajectory"],
                    trajectory_points=trajectory_points,
                    game_label=game_label,
                    custom_game_dir=self.game_catalog.custom_game_dir,
                    hide_first=hide_first,
                )
            with self._detail_figure_lock:
                if generation != self._detail_figure_generation:
                    raise RuntimeError("equilibrium convergence figure generation was invalidated")
                if any(not path.is_file() or path.stat().st_mtime_ns != mtime for path, mtime in input_state.items()):
                    raise RuntimeError("experiment group changed while equilibrium convergence figures were generated")
                for name, temporary_path in temporary_paths.items():
                    if temporary_path.is_file():
                        os.replace(temporary_path, output_paths[name])
        return output_paths

    def _invalidate_detail_figures(self) -> None:
        with self._convergence_future_lock:
            for future in self._convergence_futures.values():
                future.cancel()
            self._convergence_futures.clear()
        with self._detail_figure_lock:
            self._detail_figure_generation += 1
            if self.detail_figure_dir.exists():
                for path in self.detail_figure_dir.glob("*.png"):
                    path.unlink()

    def delete_experiment(self, filename: str) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            csv_path = self.raw_dir / filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"experiment {filename} does not exist")

            self._invalidate_detail_figures()
            csv_path.unlink()
            try:
                self._publish_plots()
            except Exception as error:
                self._clear_figure_files()
                raise PlotUpdateError(
                    f"deleted {filename}, but figure rebuilding failed; existing "
                    f"figures were cleared: {error}"
                ) from error

        self.jobs.run_maintenance(operation)

    def clear_results(self) -> None:
        def operation() -> None:
            self._invalidate_detail_figures()
            if self.raw_dir.exists():
                for path in self.raw_dir.glob("*.csv"):
                    path.unlink()
            if self.figure_dir.exists():
                for path in self.figure_dir.glob("*.png"):
                    path.unlink()

            report_path = self.results_dir / "index.html"
            if report_path.is_file():
                report_path.unlink()

        self.jobs.run_maintenance(operation)

    def result_snapshot(self) -> ResultSnapshot:
        return self.result_index.snapshot()

    def figure_records(self) -> list[dict]:
        if not self.figure_dir.exists():
            return []

        records = []
        for path in sorted(self.figure_dir.glob("*.png")):
            metadata = self._parse_figure_filename(path.name)
            if metadata is not None:
                records.append({"filename": path.name, **metadata})
        regret_order = {"external": 0, "internal": 1, "swap": 2}
        return sorted(records, key=lambda record: (record["source"], record["view"], record["player"], regret_order[record["regret"]]))

    def _parse_figure_filename(self, filename: str) -> dict | None:
        for game_name in sorted(self.games, key=len, reverse=True):
            prefix = f"{game_name}_"
            if not filename.startswith(prefix):
                continue

            remainder = filename[len(prefix):]
            average_match = re.fullmatch(
                r"average_(expected|realized)_(external|internal|swap)_"
                r"regret_player_(\d+)\.png",
                remainder,
            )
            scaling_match = re.fullmatch(
                r"(expected|realized)_(external|internal|swap)_"
                r"regret_over_sqrt_t_player_(\d+)\.png",
                remainder,
            )
            match = average_match or scaling_match
            if match is None:
                return None
            return {
                "game": game_name,
                "source": match.group(1),
                "regret": match.group(2),
                "player": int(match.group(3)),
                "view": "average" if average_match else "sqrt_scaling",
            }
        return None

    def validate_csv_filename(self, filename: str) -> str:
        return validate_leaf_filename(filename, ".csv")

    def validate_figure_filename(self, filename: str) -> str:
        filename = validate_leaf_filename(filename, ".png")
        if self._parse_figure_filename(filename) is None:
            raise ValueError("unknown figure filename")
        return filename
