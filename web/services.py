from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Lock

import numpy as np

from config import (
    ACTION_SCALING_ACTION_COUNTS,
    ADVERSARIAL_ACTIONS,
    CUSTOM_GAME_DIR,
    HORIZON,
    REPLICATES,
    SEED,
)
from experiments.cleanup import clear_experiment_artifacts
from experiments.algorithm_labels import algorithm_label, algorithm_profile_label
from experiments.game_catalog import (
    GameCatalog,
    GameDefinition,
    payoff_tensor_digest,
)
from experiments.games import PAYOFF_FACTORIES
from experiments.plots import (
    FIGURE_SUFFIXES,
    figure_pair_is_current,
    figure_paths,
    publish_figure_pair,
)
from experiments.results import iter_result_rows
from experiments.result_catalog import ResultRepository, ResultSet, ResultKind
from experiments.parallel import run_replicates
from experiments.spec import ExperimentSpec
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    ENVIRONMENT_LABELS,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    run_adversarial_experiment,
)
from experiments.scenarios.adversarial_scaling import (
    AdversarialScalingSpec,
    run_adversarial_scaling_experiment,
)
from experiments.scenarios.cross_play import ALGORITHMS_BY_FEEDBACK_MODE, FEEDBACK_MODE_LABELS, run_cross_play_experiment
from web.jobs import Job, JobContext, JobManager
from web.presentations import GAME_PRESENTATIONS
from web.validation import (
    AdversarialExperimentForm,
    ExperimentForm,
    validate_leaf_filename,
)


logger = logging.getLogger(__name__)


def _publish_figure_files(source_paths: list[Path], output_dir: Path) -> None:
    generated_names = {path.name for path in source_paths}
    for path in source_paths:
        os.replace(path, output_dir / path.name)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES and path.name not in generated_names:
            path.unlink()


def _figure_file_record(path: Path) -> dict:
    pdf_path = path.with_suffix(".pdf")
    record = {
        "filename": path.name,
        "pdf_filename": pdf_path.name if pdf_path.is_file() else None,
    }
    return record


def _validate_result_file(directory: Path, filename: str, suffix: str) -> str:
    filename = validate_leaf_filename(filename, suffix)
    if not (directory / filename).is_file():
        raise FileNotFoundError(filename)
    return filename


def _validate_result_figure(directory: Path, filename: str, records: list[dict]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in FIGURE_SUFFIXES:
        raise ValueError("invalid figure filename")
    filename = validate_leaf_filename(filename, suffix)
    preview_name = Path(filename).with_suffix(".png").name
    known = any(
        preview_name == record["filename"]
        for record in records
    )
    if not known or not (directory / filename).is_file():
        raise FileNotFoundError(filename)
    return filename


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
        replicate_workers: int | None = None,
    ):
        self.results_dir = Path(results_dir)
        self.raw_dir = Path(raw_dir)
        self.figure_dir = Path(figure_dir)
        self.adversarial_dir = self.results_dir / "adversarial"
        self.adversarial_raw_dir = self.adversarial_dir / "raw"
        self.adversarial_scaling_dir = self.adversarial_dir / "scaling"
        self.adversarial_scaling_raw_dir = self.adversarial_scaling_dir / "raw"
        self.adversarial_scaling_figure_dir = self.adversarial_scaling_dir / "figures"
        self.game_catalog = GameCatalog(custom_game_dir)
        self.jobs = job_manager or JobManager()
        self.replicate_workers = replicate_workers
        self.results = {kind: ResultRepository(directory, kind) for kind, directory in (
            ("fixed", self.raw_dir), ("adversarial", self.adversarial_raw_dir), ("scaling", self.adversarial_scaling_raw_dir))}
        from web.figure_builder import FigureBuilder

        self.figure_builder = FigureBuilder(self)
        self._detail_figure_lock = Lock()
        self._detail_figure_generation = 0
        self._convergence_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="equilibrium-distance",
        )
        self._convergence_future_lock = Lock()
        self._convergence_futures: dict[str, Future[Path]] = {}

    def _clear_generated_artifacts(self, roots: tuple[Path, ...]) -> tuple[Path, ...]:
        return clear_experiment_artifacts(
            roots,
            preserve=(self.game_catalog.custom_game_dir,),
        )

    def _clear_experiment_caches(self) -> None:
        self._clear_generated_artifacts((
            self.results_dir / "cache",
            self.adversarial_dir / "cache",
        ))

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
        if game_name in PAYOFF_FACTORIES:
            return True
        definition = self.game_definitions.get(game_name)
        return bool(
            definition is not None
            and definition.source == "custom"
            and definition.n_players == 2
            and definition.payoff_structure == "zero_sum"
        )

    def supports_equilibrium_distance(self, game_name: str) -> bool:
        return game_name in self.game_definitions

    def custom_games(self) -> tuple[list[GameDefinition], list[str]]:
        return self.game_catalog.custom_definitions()

    def create_custom_game(
        self,
        name: str,
        n_players,
        action_counts,
        seed,
        payoff_structure: str = "general_sum",
    ) -> GameDefinition:
        return self.game_catalog.create_random(
            name,
            n_players,
            action_counts,
            seed,
            payoff_structure,
        )

    def delete_custom_game(self, game_id: str) -> GameDefinition:
        def operation() -> GameDefinition:
            result_prefix = f"{game_id}_"
            if self.raw_dir.exists() and any(path.name.startswith(result_prefix) for path in self.raw_dir.glob("*.csv")):
                raise ValueError("delete the recorded experiments for this game before deleting the game")
            definition = self.game_catalog.delete(game_id)
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
    def feedback_modes(self) -> dict[str, str]:
        return dict(FEEDBACK_MODE_LABELS)

    @property
    def algorithms_by_feedback_mode(self) -> dict[str, list[str]]:
        return {
            name: list(algorithms)
            for name, algorithms in ALGORITHMS_BY_FEEDBACK_MODE.items()
        }

    @property
    def adversarial_algorithms_by_feedback_mode(self) -> dict[str, list[str]]:
        return self.algorithms_by_feedback_mode

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
            "algorithm_names": [first_algorithm] * self.game_player_counts[game],
            "horizon": HORIZON,
            "seed": SEED,
            "replicates": REPLICATES,
        }

    def default_adversarial_form_state(self) -> dict:
        feedback_mode = "full_information"
        first_algorithm = self.adversarial_algorithms_by_feedback_mode[feedback_mode][0]
        return {
            "environment": HISTORICAL_FREQUENCY_ENVIRONMENT,
            "feedback_mode": feedback_mode,
            "algorithm_names": [first_algorithm],
            "n_actions": ADVERSARIAL_ACTIONS,
            "horizon": HORIZON,
            "environment_seed": SEED,
            "seed": SEED,
            "replicates": REPLICATES,
            "scaling_action_counts": ", ".join(map(str, ACTION_SCALING_ACTION_COUNTS)),
            "scaling_replicates": REPLICATES,
        }

    def _submit_replicates(
        self,
        specs: list,
        raw_dir: Path,
        resource_key: Callable,
        description: str,
        run: Callable,
        task_kwargs: Callable,
        duplicate_message: str,
    ) -> Job:
        reserved = self.jobs.reserved_resources()
        missing = [
            spec
            for spec in specs
            if resource_key(spec) not in reserved and not (raw_dir / f"{spec.run_id}.csv").exists()
        ]
        if not missing:
            raise FileExistsError(duplicate_message)

        def operation(job: JobContext) -> str:
            run_replicates(
                run, [task_kwargs(spec) for spec in missing],
                workers=self.replicate_workers,
                should_cancel=lambda: job.cancelled, completed=job.advance,
            )
            job.check_cancelled()
            return f"Completed {len(missing)} run(s); skipped {len(specs) - len(missing)} existing or queued"

        return self.jobs.submit(
            description,
            operation,
            total=len(missing),
            resource_keys={resource_key(spec) for spec in missing},
        )

    def submit_adversarial_experiment(
        self,
        form: AdversarialExperimentForm,
    ) -> Job:
        specs = [
            AdversarialExperimentSpec(
                environment=form.environment,
                environment_seed=form.environment_seed,
                feedback_mode=form.feedback_mode,
                algorithm_name=form.algorithm_name,
                n_actions=form.n_actions,
                horizon=form.horizon,
                seed=form.learner_seed,
                replicate=replicate,
            )
            for replicate in range(form.replicates)
        ]
        def task_kwargs(spec):
            return dict(
                environment=spec.environment,
                environment_seed=spec.environment_seed,
                feedback_mode=spec.feedback_mode,
                algorithm_name=spec.algorithm_name,
                n_actions=spec.n_actions,
                horizon=spec.horizon,
                seed=spec.seed,
                replicate=spec.replicate,
                output_dir=self.adversarial_raw_dir,
            )

        return self._submit_replicates(
            specs,
            self.adversarial_raw_dir,
            lambda spec: f"adversarial:{spec.run_id}",
            (
                f"Adversarial: {algorithm_label(form.algorithm_name)} · "
                f"{ENVIRONMENT_LABELS[form.environment]} · "
                f"{FEEDBACK_MODE_LABELS[form.feedback_mode]} · "
                f"{form.n_actions} actions · "
                f"{form.replicates} replicates · base learner seed {form.learner_seed}"
            ),
            run_adversarial_experiment,
            task_kwargs,
            "all requested adversarial replicates already exist or are queued",
        )

    def submit_adversarial_scaling_experiment(
        self,
        spec: AdversarialScalingSpec,
    ) -> Job:
        resource_key = f"adversarial-scaling:{spec.run_id}"
        if resource_key in self.jobs.reserved_resources() or (
            self.adversarial_scaling_raw_dir / f"{spec.run_id}.csv"
        ).exists():
            raise FileExistsError(
                "the requested action-space scaling experiment already exists or is queued"
            )

        def operation(job: JobContext) -> str:
            run_adversarial_scaling_experiment(
                spec,
                self.adversarial_scaling_raw_dir,
                should_cancel=lambda: job.cancelled,
                completed=job.advance,
                workers=self.replicate_workers,
            )
            job.check_cancelled()
            try:
                self._publish_adversarial_scaling_plots()
            except Exception as error:
                raise PlotUpdateError(
                    "action-space scaling results were saved, but their figures "
                    f"could not be rebuilt: {error}"
                ) from error
            return (
                f"Completed {len(spec.action_counts)} action counts × "
                f"{spec.replicates} replicates"
            )

        return self.jobs.submit(
            f"Action scaling: {algorithm_label(spec.algorithm_name)} · "
            f"{ENVIRONMENT_LABELS[spec.environment]}",
            operation,
            total=len(spec.action_counts) * spec.replicates,
            resource_keys={resource_key},
        )

    def _publish_adversarial_scaling_plots(self) -> None:
        from experiments.plots.plot_adversarial_scaling import (
            plot_adversarial_scaling_results,
        )

        self._publish_generated_plots(
            self.adversarial_scaling_raw_dir,
            self.adversarial_scaling_figure_dir,
            ".action-scaling-figures-",
            plot_adversarial_scaling_results,
        )

    @staticmethod
    def _publish_generated_plots(
        raw_dir: Path,
        figure_dir: Path,
        prefix: str,
        plotter: Callable[..., object],
    ) -> None:
        parent_dir = figure_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=prefix,
            dir=parent_dir,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            plotter(raw_dir, temporary_path, skip_invalid=True)
            generated_paths = [
                path
                for path in temporary_path.iterdir()
                if path.suffix.lower() in FIGURE_SUFFIXES
            ]
            _publish_figure_files(generated_paths, figure_dir)

    def _delete_result(
        self,
        directory: Path,
        figure_directory: Path,
        filename: str,
        rebuild: Callable[[], None],
    ) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(filename)

            directory.parent.mkdir(parents=True, exist_ok=True)
            backup_directory = Path(
                tempfile.mkdtemp(prefix=".delete-result-", dir=directory.parent)
            )
            backup_csv = backup_directory / path.name
            backup_figures = backup_directory / "figures"
            csv_moved = False
            figures_moved = False
            rebuild_started = False
            try:
                os.replace(path, backup_csv)
                csv_moved = True
                if figure_directory.exists():
                    os.replace(figure_directory, backup_figures)
                    figures_moved = True
                rebuild_started = True
                rebuild()
                self._clear_experiment_caches()
            except Exception as error:
                try:
                    if rebuild_started:
                        if figure_directory.is_dir():
                            shutil.rmtree(figure_directory)
                        elif figure_directory.exists():
                            figure_directory.unlink()
                    if figures_moved and backup_figures.exists():
                        os.replace(backup_figures, figure_directory)
                    if csv_moved and backup_csv.exists():
                        os.replace(backup_csv, path)
                except Exception as restore_error:
                    raise PlotUpdateError(
                        f"could not delete {filename}; rollback also failed and "
                        f"recoverable files remain in {backup_directory}: "
                        f"{restore_error}"
                    ) from error
                shutil.rmtree(backup_directory, ignore_errors=True)
                raise PlotUpdateError(
                    f"could not delete {filename}; the CSV and previous figures "
                    f"were restored: {error}"
                ) from error
            else:
                shutil.rmtree(backup_directory, ignore_errors=True)

        self.jobs.run_maintenance(operation)

    def adversarial_scaling_figure_records(self, results: ResultSet | None = None) -> list[dict]:
        if results is None:
            results = self.result_snapshot("scaling")
        records = []
        for result in results.records:
            path = self.adversarial_scaling_figure_dir / (
                f"{result.run_id}_regret_by_actions.png"
            )
            if not path.is_file():
                continue
            records.append(
                {
                    **result.summary(),
                    **_figure_file_record(path),
                }
            )
        return records

    def validate_adversarial_scaling_csv_filename(self, filename: str) -> str:
        return _validate_result_file(self.adversarial_scaling_raw_dir, filename, ".csv")

    def validate_adversarial_scaling_figure_filename(self, filename: str) -> str:
        return _validate_result_figure(
            self.adversarial_scaling_figure_dir,
            filename,
            self.adversarial_scaling_figure_records(),
        )

    def delete_adversarial_scaling_experiment(self, filename: str) -> None:
        self._delete_result(
            self.adversarial_scaling_raw_dir,
            self.adversarial_scaling_figure_dir,
            filename,
            self._publish_adversarial_scaling_plots,
        )

    def validate_adversarial_csv_filename(self, filename: str) -> str:
        return _validate_result_file(self.adversarial_raw_dir, filename, ".csv")

    def delete_adversarial_experiment(self, filename: str) -> None:
        self._delete_ordinary_result(self.adversarial_raw_dir, filename)

    def clear_adversarial_results(self) -> tuple[int, int]:
        def operation() -> tuple[int, int]:
            generated = (
                [path for path in self.adversarial_dir.rglob("*") if path.is_file()]
                if self.adversarial_dir.exists()
                else []
            )
            csv_count = sum(path.suffix.lower() == ".csv" for path in generated)
            figure_count = sum(path.suffix.lower() == ".png" for path in generated)
            self._clear_generated_artifacts((
                self.adversarial_dir,
                self.results_dir / "cache",
            ))
            return csv_count, figure_count

        return self.jobs.run_maintenance(operation)

    def _spec(self, form: ExperimentForm, replicate: int) -> ExperimentSpec:
        return ExperimentSpec(
            game_name=form.game,
            feedback_mode=form.feedback_mode,
            algorithm_names=form.algorithm_names,
            horizon=form.horizon,
            seed=form.seed,
            replicate=replicate,
            game_payoff_digest=payoff_tensor_digest(self.game_catalog.load(form.game)),
        )

    def submit_experiment(self, form: ExperimentForm) -> Job:
        specs = [self._spec(form, replicate=replicate) for replicate in range(form.replicates)]

        def task_kwargs(spec):
            return dict(
                game_name=spec.game_name,
                feedback_mode=spec.feedback_mode,
                algorithm_names=list(spec.algorithm_names),
                horizon=spec.horizon,
                seed=spec.seed,
                replicate=spec.replicate,
                output_dir=self.raw_dir,
                custom_game_dir=self.game_catalog.custom_game_dir,
            )

        return self._submit_replicates(
            specs,
            self.raw_dir,
            lambda spec: spec.run_id,
            f"{form.game}: {algorithm_profile_label(form.algorithm_names)}",
            run_cross_play_experiment,
            task_kwargs,
            "all requested replicates already exist or are queued",
        )

    @property
    def detail_figure_dir(self) -> Path:
        return self.figure_dir / "details"

    def _result_group_paths(self, group_id: str) -> list[Path]:
        if re.fullmatch(r"[0-9a-f]{16}", group_id) is None:
            raise ValueError("invalid result group")
        paths = self.result_snapshot().detail_paths(group_id)
        for path in paths:
            validate_leaf_filename(path.name, ".csv")
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(group_id)
        return paths

    def _group_cache_stem(self, group_id: str, input_paths: list[Path]) -> str:
        membership = "\n".join(path.name for path in input_paths)
        return f"{group_id}_{sha256(membership.encode('utf-8')).hexdigest()[:8]}"

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
            if figure_pair_is_current(output_path, input_mtime):
                return output_path

            from experiments.plots.plot_joint_actions import plot_joint_actions

            self.detail_figure_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".joint-actions-", dir=self.detail_figure_dir) as temporary_directory:
                temporary_path = Path(temporary_directory) / output_path.name
                plot_joint_actions(
                    input_paths,
                    temporary_path,
                    self.game_catalog.custom_game_dir,
                )
                publish_figure_pair(temporary_path, output_path)
        return output_path

    def _convergence_figure_path(
        self,
        filename: str,
    ) -> tuple[Path, Path]:
        from experiments.plots.plot_equilibrium_convergence import EQUILIBRIUM_DISTANCE_FIGURE_VERSION

        filename = validate_leaf_filename(filename, ".csv")
        input_path = self.raw_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(filename)
        game_name = next(iter_result_rows(input_path))["game"]
        if not self.supports_equilibrium_distance(game_name):
            raise ValueError(f"equilibrium distance is unavailable for {game_name}")
        return (
            input_path,
            self.detail_figure_dir
            / f"{input_path.stem}_v{EQUILIBRIUM_DISTANCE_FIGURE_VERSION}_equilibrium_distance.png",
        )

    def _group_convergence_figure_path(
        self,
        group_id: str,
    ) -> tuple[list[Path], Path, str]:
        from experiments.plots.plot_equilibrium_convergence import EQUILIBRIUM_DISTANCE_FIGURE_VERSION

        input_paths = self._result_group_paths(group_id)
        game_name = next(iter_result_rows(input_paths[0]))["game"]
        if not self.supports_equilibrium_distance(game_name):
            raise ValueError(f"equilibrium distance is unavailable for {game_name}")
        cache_stem = self._group_cache_stem(group_id, input_paths)
        return (
            input_paths,
            self.detail_figure_dir
            / f"{cache_stem}_v{EQUILIBRIUM_DISTANCE_FIGURE_VERSION}_replicate_mean_equilibrium_distance.png",
            cache_stem,
        )

    def _request_convergence_figure(
        self,
        input_paths: list[Path],
        output_path: Path,
        future_key: str,
        generate: Callable[[], Path],
        log_context: str,
    ) -> tuple[Path | None, str | None]:
        input_mtime = max(path.stat().st_mtime_ns for path in input_paths)
        if figure_pair_is_current(output_path, input_mtime):
            with self._convergence_future_lock:
                future = self._convergence_futures.get(future_key)
                if future is not None and future.done():
                    self._convergence_futures.pop(future_key, None)
            return output_path, None

        scheduled = False
        with self._convergence_future_lock:
            future = self._convergence_futures.get(future_key)
            if future is None:
                future = self._convergence_executor.submit(generate)
                self._convergence_futures[future_key] = future
                scheduled = True
        if scheduled or not future.done():
            return None, None

        with self._convergence_future_lock:
            self._convergence_futures.pop(future_key, None)
        try:
            return future.result(), None
        except Exception as error:
            logger.exception(
                "Equilibrium distance generation failed for %s",
                log_context,
            )
            return None, f"{type(error).__name__}: {error}"

    def request_equilibrium_convergence_figure(
        self,
        filename: str,
    ) -> tuple[Path | None, str | None]:
        input_path, output_path = self._convergence_figure_path(filename)
        return self._request_convergence_figure(
            [input_path],
            output_path,
            filename,
            lambda: self._generate_equilibrium_distance(
                [input_path],
                output_path,
            ),
            filename,
        )

    def request_group_equilibrium_convergence_figure(
        self,
        group_id: str,
    ) -> tuple[Path | None, str | None]:
        input_paths, output_path, cache_stem = self._group_convergence_figure_path(
            group_id
        )
        return self._request_convergence_figure(
            input_paths,
            output_path,
            f"group:{cache_stem}",
            lambda: self._generate_equilibrium_distance(
                input_paths,
                output_path,
            ),
            f"group {group_id}",
        )

    def _generate_equilibrium_distance(
        self,
        input_paths: list[Path],
        output_path: Path,
    ) -> Path:
        with self._detail_figure_lock:
            input_state = {path: path.stat().st_mtime_ns for path in input_paths}
            input_mtime = max(input_state.values())
            if figure_pair_is_current(output_path, input_mtime):
                return output_path
            generation = self._detail_figure_generation
            self.detail_figure_dir.mkdir(parents=True, exist_ok=True)

        from experiments.plots.plot_equilibrium_convergence import (
            plot_result_equilibrium_distance,
        )

        with tempfile.TemporaryDirectory(
            prefix=".equilibrium-convergence-",
            dir=self.detail_figure_dir,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory) / output_path.name
            plot_result_equilibrium_distance(
                input_paths,
                temporary_path,
                custom_game_dir=self.game_catalog.custom_game_dir,
                cache_dir=self.results_dir / "cache" / "equilibrium_distance",
            )
            with self._detail_figure_lock:
                if generation != self._detail_figure_generation:
                    raise RuntimeError("equilibrium convergence figure generation was invalidated")
                if any(not path.is_file() or path.stat().st_mtime_ns != mtime for path, mtime in input_state.items()):
                    raise RuntimeError("experiment group changed while equilibrium convergence figures were generated")
                publish_figure_pair(temporary_path, output_path)
        return output_path

    def _invalidate_detail_figures(self) -> None:
        with self._convergence_future_lock:
            for future in self._convergence_futures.values():
                future.cancel()
            self._convergence_futures.clear()
        with self._detail_figure_lock:
            self._detail_figure_generation += 1
            self._clear_generated_artifacts((self.detail_figure_dir,))

    def delete_experiment(self, filename: str) -> None:
        self._delete_ordinary_result(self.raw_dir, filename)

    def _delete_ordinary_result(self, directory: Path, filename: str) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            csv_path = directory / filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"experiment {filename} does not exist")

            if directory == self.raw_dir:
                self._invalidate_detail_figures()
            self._clear_experiment_caches()
            csv_path.unlink()

        self.jobs.run_maintenance(operation)

    def clear_results(self) -> None:
        def operation() -> None:
            self._invalidate_detail_figures()
            self._clear_generated_artifacts((
                self.raw_dir,
                self.figure_dir,
                self.adversarial_dir,
                self.results_dir / "cache",
            ))

        self.jobs.run_maintenance(operation)

    def result_snapshot(self, kind: ResultKind = "fixed") -> ResultSet:
        return self.results[kind].snapshot(self.game_definitions if kind == "fixed" else None)

    def validate_csv_filename(self, filename: str) -> str:
        return validate_leaf_filename(filename, ".csv")
