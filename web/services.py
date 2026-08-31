from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
import csv
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
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
from experiments.algorithm_labels import algorithm_label, algorithm_profile_label
from experiments.game_catalog import (
    CUSTOM_GAME_PREFIX,
    GameCatalog,
    GameDefinition,
    payoff_tensor_digest,
)
from experiments.games import PAYOFF_FACTORIES
from experiments.plots import (
    FIGURE_SUFFIXES,
    confidence_free_figure_path,
    figure_pair_is_current,
    figure_path,
    figure_paths,
    publish_figure_pair,
)
from experiments.results import iter_result_rows
from experiments.spec import ExperimentSpec
from experiments.scenarios.adversarial import (
    ALGORITHMS_BY_FEEDBACK_MODE as ADVERSARIAL_ALGORITHMS_BY_FEEDBACK_MODE,
    AdversarialExperimentSpec,
    ENVIRONMENT_LABELS,
    FEEDBACK_MODE_LABELS,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    TARGET_REGRET_BY_ALGORITHM,
    adversarial_environment_detail,
    load_final_adversarial_row,
    run_adversarial_experiment,
)
from experiments.result_schema import regret_sources
from experiments.scenarios.adversarial_scaling import (
    AdversarialScalingSpec,
    adversarial_scaling_environment_detail,
    load_adversarial_scaling_rows,
    run_adversarial_scaling_experiment,
)
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.experiment_modes import FEEDBACK_MODES
from web.jobs import Job, JobContext, JobManager
from web.presentations import GAME_PRESENTATIONS
from web.result_groups import (
    result_group_filenames,
)
from web.result_index import ResultIndex, ResultSnapshot
from web.validation import (
    AdversarialExperimentForm,
    AdversarialScalingForm,
    ExperimentForm,
    validate_leaf_filename,
)


logger = logging.getLogger(__name__)


def _publish_figure_files(source_paths: list[Path], output_dir: Path, filename_prefix: str | None = None) -> None:
    generated_names = {path.name for path in source_paths}
    for path in source_paths:
        os.replace(path, output_dir / path.name)
    for path in output_dir.iterdir():
        matches_prefix = filename_prefix is None or path.name.startswith(filename_prefix)
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES and matches_prefix and path.name not in generated_names:
            path.unlink()


def _figure_file_record(path: Path, confidence_intervals: bool = False) -> dict:
    pdf_path = path.with_suffix(".pdf")
    record = {
        "filename": path.name,
        "pdf_filename": pdf_path.name if pdf_path.is_file() else None,
    }
    if confidence_intervals:
        confidence_free_path = confidence_free_figure_path(path)
        if confidence_free_path.is_file():
            confidence_free_pdf_path = confidence_free_path.with_suffix(".pdf")
            record.update(
                confidence_free_filename=confidence_free_path.name,
                confidence_free_pdf_filename=(confidence_free_pdf_path.name if confidence_free_pdf_path.is_file() else record["pdf_filename"]),
            )
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
        preview_name in {record["filename"], record.get("confidence_free_filename")}
        for record in records
    )
    if not known or not (directory / filename).is_file():
        raise FileNotFoundError(filename)
    return filename


def _clear_result_files(raw_dirs: tuple[Path, ...], figure_dirs: tuple[Path, ...]) -> tuple[int, int]:
    csv_paths = [path for directory in raw_dirs if directory.exists() for path in directory.glob("*.csv")]
    figure_paths = [
        path
        for directory in figure_dirs
        if directory.exists()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES
    ]
    for path in csv_paths + figure_paths:
        path.unlink()
    return len(csv_paths), sum(path.suffix.lower() == ".png" for path in figure_paths)


def _load_summaries(directory: Path, summarize: Callable[[Path], dict]) -> tuple[list[dict], list[str]]:
    summaries = []
    warnings = []
    for path in sorted(directory.glob("*.csv")):
        try:
            summaries.append(summarize(path))
        except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
            warnings.append(f"Skipped {path.name}: {error}")
    return summaries, warnings


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
        self.adversarial_dir = self.results_dir / "adversarial"
        self.adversarial_raw_dir = self.adversarial_dir / "raw"
        self.adversarial_figure_dir = self.adversarial_dir / "figures"
        self.adversarial_scaling_dir = self.adversarial_dir / "scaling"
        self.adversarial_scaling_raw_dir = self.adversarial_scaling_dir / "raw"
        self.adversarial_scaling_figure_dir = self.adversarial_scaling_dir / "figures"
        self.game_catalog = GameCatalog(custom_game_dir)
        self.jobs = job_manager or JobManager()
        self.result_index = ResultIndex(self.raw_dir)
        self._experimental_trajectory_dashboard = None
        self._detail_figure_lock = Lock()
        self._equilibrium_figure_lock = Lock()
        self._detail_figure_generation = 0
        self._convergence_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="equilibrium-distance",
        )
        self._convergence_future_lock = Lock()
        self._convergence_futures: dict[str, Future[Path]] = {}

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

    @property
    def experimental_trajectory(self):
        """Load the opt-in trajectory service only when explicitly used."""
        if self._experimental_trajectory_dashboard is None:
            from experimental.equilibrium_trajectory.dashboard import (
                ExperimentalTrajectoryDashboard,
            )

            self._experimental_trajectory_dashboard = (
                ExperimentalTrajectoryDashboard(self)
            )
        return self._experimental_trajectory_dashboard

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
            self._clear_figure_files(game_id)
            self._clear_custom_equilibrium_figures(game_id)
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
        return {name: mode.label for name, mode in FEEDBACK_MODES.items()}

    @property
    def algorithms_by_feedback_mode(self) -> dict[str, list[str]]:
        return {
            name: list(mode.algorithms)
            for name, mode in FEEDBACK_MODES.items()
        }

    @property
    def adversarial_algorithms_by_feedback_mode(self) -> dict[str, list[str]]:
        return {
            mode: list(algorithms)
            for mode, algorithms in ADVERSARIAL_ALGORITHMS_BY_FEEDBACK_MODE.items()
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
            "replicates": REPLICATES,
        }

    def default_adversarial_form_state(self) -> dict:
        feedback_mode = "full_information"
        first_algorithm = self.adversarial_algorithms_by_feedback_mode[feedback_mode][0]
        return {
            "environment": HISTORICAL_FREQUENCY_ENVIRONMENT,
            "initialization_mode": "centered",
            "feedback_mode": feedback_mode,
            "regret_evaluation": "both",
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
        rebuild: Callable[[], None],
        duplicate_message: str,
        rebuild_error: str,
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
            for spec in missing:
                job.check_cancelled()
                run(spec, job)
                job.advance()
            job.check_cancelled()
            try:
                rebuild()
            except Exception as error:
                raise PlotUpdateError(f"{rebuild_error}: {error}") from error
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
                initialization_mode=form.initialization_mode,
                environment_seed=form.environment_seed,
                feedback_mode=form.feedback_mode,
                algorithm_name=form.algorithm_name,
                n_actions=form.n_actions,
                horizon=form.horizon,
                seed=form.learner_seed,
                replicate=replicate,
                regret_evaluation=form.regret_evaluation,
            )
            for replicate in range(form.replicates)
        ]
        def run(spec, job):
            run_adversarial_experiment(
                environment=spec.environment,
                initialization_mode=spec.initialization_mode,
                environment_seed=spec.environment_seed,
                feedback_mode=spec.feedback_mode,
                algorithm_name=spec.algorithm_name,
                n_actions=spec.n_actions,
                horizon=spec.horizon,
                seed=spec.seed,
                replicate=spec.replicate,
                regret_evaluation=spec.regret_evaluation,
                output_dir=self.adversarial_raw_dir,
                should_cancel=lambda: job.cancelled,
            )

        return self._submit_replicates(
            specs,
            self.adversarial_raw_dir,
            lambda spec: f"adversarial:{spec.run_id}",
            (
                f"Adversarial: {algorithm_label(form.algorithm_name)} · "
                f"{ENVIRONMENT_LABELS[form.environment]} · "
                f"{FEEDBACK_MODE_LABELS[form.feedback_mode]} · "
                f"{form.regret_evaluation} regret · "
                f"{form.n_actions} actions · "
                f"{form.replicates} replicates · base learner seed {form.learner_seed}"
            ),
            run,
            self._publish_adversarial_plots,
            "all requested adversarial replicates already exist or are queued",
            "adversarial runs were saved, but their figures could not be rebuilt",
        )

    def submit_adversarial_scaling_experiment(
        self,
        form: AdversarialScalingForm,
    ) -> Job:
        spec = AdversarialScalingSpec(
            environment=form.environment,
            initialization_mode=form.initialization_mode,
            feedback_mode=form.feedback_mode,
            algorithm_name=form.algorithm_name,
            action_counts=form.action_counts,
            replicates=form.replicates,
            horizon=form.horizon,
            environment_seed=form.environment_seed,
            learner_seed=form.learner_seed,
            regret_evaluation=form.regret_evaluation,
        )
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
            f"Action scaling: {algorithm_label(form.algorithm_name)} · "
            f"{ENVIRONMENT_LABELS[form.environment]}",
            operation,
            total=len(spec.action_counts) * spec.replicates,
            resource_keys={resource_key},
        )

    def _publish_adversarial_plots(self) -> None:
        from experiments.plots.plot_adversarial import plot_adversarial_results

        self._publish_generated_plots(
            self.adversarial_raw_dir,
            self.adversarial_figure_dir,
            ".adversarial-figures-",
            plot_adversarial_results,
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
        filename_prefix: str | None = None,
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
            _publish_figure_files(generated_paths, figure_dir, filename_prefix)

    def _delete_result(self, directory: Path, filename: str, rebuild: Callable[[], None]) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(filename)
            path.unlink()
            rebuild()

        self.jobs.run_maintenance(operation)

    def adversarial_scaling_summaries(self) -> tuple[list[dict], list[str]]:
        def summarize(path: Path) -> dict:
            first = load_adversarial_scaling_rows(path)[0]
            return {
                "filename": path.name,
                "run_id": first["run_id"],
                "environment": first["environment"],
                "environment_label": ENVIRONMENT_LABELS[first["environment"]],
                "environment_detail": adversarial_scaling_environment_detail(first),
                "feedback_label": FEEDBACK_MODE_LABELS[first["feedback_mode"]],
                "regret_evaluation": first["regret_evaluation"],
                "implementation_version": int(first.get("implementation_version", 0)),
                "algorithm_label": algorithm_label(first["algorithm"]),
                "action_counts": [int(value) for value in first["action_counts"].split(",")],
                "replicates": int(first["replicates"]),
                "horizon": int(first["horizon"]),
                "base_learner_seed": int(first["base_learner_seed"]),
                "target_regret": first["target_regret"],
            }

        return _load_summaries(self.adversarial_scaling_raw_dir, summarize)

    def adversarial_scaling_figure_records(self, summaries: list[dict] | None = None) -> list[dict]:
        if summaries is None:
            summaries, _ = self.adversarial_scaling_summaries()
        records = []
        for summary in summaries:
            for source in regret_sources(summary["regret_evaluation"]):
                path = self.adversarial_scaling_figure_dir / (
                    f"{summary['run_id']}_{source}_regret_by_actions.png"
                )
                if not path.is_file():
                    continue
                records.append(
                    {
                        **summary,
                        **_figure_file_record(path, confidence_intervals=True),
                        "source": source,
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
        self._delete_result(self.adversarial_scaling_raw_dir, filename, self._publish_adversarial_scaling_plots)

    def adversarial_result_summaries(self) -> tuple[list[dict], list[str]]:
        def summarize(path: Path) -> dict:
            row = load_final_adversarial_row(path)
            algorithm = row["algorithm"]
            target_regret = TARGET_REGRET_BY_ALGORITHM.get(algorithm, "external")
            sources = regret_sources(row["regret_evaluation"])
            return {
                "filename": path.name,
                "algorithm": algorithm,
                "algorithm_label": algorithm_label(algorithm),
                "feedback_mode": row["feedback_mode"],
                "feedback_label": FEEDBACK_MODE_LABELS[row["feedback_mode"]],
                "regret_evaluation": row["regret_evaluation"],
                "implementation_version": int(row["implementation_version"]),
                "environment": row["environment"],
                "environment_label": ENVIRONMENT_LABELS[row["environment"]],
                "environment_detail": adversarial_environment_detail(row),
                "n_actions": int(row["n_actions"]),
                "horizon": int(row["horizon"]),
                "environment_seed": int(row["environment_seed"]) if row["environment_seed"] else None,
                "learner_seed": int(row["learner_seed"]),
                "replicate": int(row["replicate"]),
                "target_regret": target_regret,
                "expected_regret": float(row[f"average_expected_{target_regret}_regret"]) if "expected" in sources else None,
                "realized_regret": float(row[f"average_realized_{target_regret}_regret"]) if "realized" in sources else None,
            }

        return _load_summaries(self.adversarial_raw_dir, summarize)

    def adversarial_figure_records(self) -> list[dict]:
        records = []
        regret_pattern = re.compile(
            r"adversarial_(.+?)_(full_information|bandit)_(\d+)_actions_(average_)?"
            r"(expected|realized)_(external|internal|swap)_regret"
            r"(_over_sqrt_t)?\.png"
        )
        for path in sorted(self.adversarial_figure_dir.glob("*.png")):
            match = regret_pattern.fullmatch(path.name)
            if match is None:
                continue
            environment = match.group(1)
            if environment not in ENVIRONMENT_LABELS:
                continue
            average = match.group(4) is not None
            scaled = match.group(7) is not None
            if average == scaled:
                continue
            records.append({
                **_figure_file_record(path, confidence_intervals=True),
                "environment": environment,
                "environment_label": ENVIRONMENT_LABELS[environment],
                "feedback_mode": match.group(2),
                "feedback_label": FEEDBACK_MODE_LABELS[match.group(2)],
                "n_actions": int(match.group(3)),
                "source": match.group(5),
                "regret": match.group(6),
                "view": "average" if average else "sqrt_scaling",
            })
        regret_order = {"external": 0, "internal": 1, "swap": 2}
        view_order = {"average": 0, "sqrt_scaling": 1}
        environment_order = {
            environment: index
            for index, environment in enumerate(ENVIRONMENT_LABELS)
        }
        feedback_order = {
            feedback: index
            for index, feedback in enumerate(FEEDBACK_MODE_LABELS)
        }
        return sorted(
            records,
            key=lambda record: (
                environment_order[record["environment"]],
                feedback_order[record["feedback_mode"]],
                record["source"],
                view_order[record["view"]],
                record["n_actions"],
                regret_order[record["regret"]],
                record["filename"],
            ),
        )

    def validate_adversarial_csv_filename(self, filename: str) -> str:
        return _validate_result_file(self.adversarial_raw_dir, filename, ".csv")

    def validate_adversarial_figure_filename(self, filename: str) -> str:
        return _validate_result_figure(
            self.adversarial_figure_dir,
            filename,
            self.adversarial_figure_records(),
        )

    def delete_adversarial_experiment(self, filename: str) -> None:
        self._delete_result(self.adversarial_raw_dir, filename, self._publish_adversarial_plots)

    def clear_adversarial_results(self) -> tuple[int, int]:
        return self.jobs.run_maintenance(
            lambda: _clear_result_files(
                (self.adversarial_raw_dir, self.adversarial_scaling_raw_dir),
                (self.adversarial_figure_dir, self.adversarial_scaling_figure_dir),
            )
        )

    def _spec(self, form: ExperimentForm, replicate: int) -> ExperimentSpec:
        return ExperimentSpec(
            game_name=form.game,
            feedback_mode=form.feedback_mode,
            algorithm_names=form.algorithm_names,
            horizon=form.horizon,
            seed=form.seed,
            replicate=replicate,
            regret_evaluation=form.regret_evaluation,
            game_payoff_digest=payoff_tensor_digest(self.game_catalog.load(form.game)),
        )

    def submit_experiment(self, form: ExperimentForm) -> Job:
        mode = FEEDBACK_MODES[form.feedback_mode]
        specs = [self._spec(form, replicate=replicate) for replicate in range(form.replicates)]

        def run(spec, job):
            mode.runner(
                game_name=spec.game_name,
                algorithm_names=list(spec.algorithm_names),
                horizon=spec.horizon,
                seed=spec.seed,
                replicate=spec.replicate,
                output_dir=self.raw_dir,
                should_cancel=lambda: job.cancelled,
                custom_game_dir=self.game_catalog.custom_game_dir,
                regret_evaluation=spec.regret_evaluation,
            )

        return self._submit_replicates(
            specs,
            self.raw_dir,
            lambda spec: spec.run_id,
            f"{form.game}: {algorithm_profile_label(form.algorithm_names)}",
            run,
            lambda: self._publish_plots(form.game),
            "all requested replicates already exist or are queued",
            "experiments were saved, but their figures could not be rebuilt",
        )

    def submit_plot_rebuild(self) -> Job:
        return self.jobs.submit(
            "Rebuild all figures",
            lambda job: self._rebuild_all_plots(job),
        )

    def _rebuild_all_plots(self, job: JobContext) -> str:
        job.check_cancelled()
        self._publish_plots()
        self._publish_adversarial_plots()
        self._publish_adversarial_scaling_plots()
        job.advance()
        return "Rebuilt all figures"

    def _publish_plots(self, game_name: str | None = None) -> None:
        from experiments.plots.plot_regret import (
            plot_all_results,
            plot_selected_results,
        )

        plotter = plot_all_results if game_name is None else lambda input_dir, output_dir, skip_invalid: plot_selected_results(game_name, input_dir, output_dir, skip_invalid)
        self._publish_generated_plots(
            self.raw_dir,
            self.figure_dir,
            ".figures-",
            plotter,
            None if game_name is None else f"{game_name}_",
        )

    def _clear_figure_files(self, game_name: str | None = None) -> None:
        if not self.figure_dir.exists():
            return
        for path in self.figure_dir.iterdir():
            matches_game = game_name is None or path.name.startswith(f"{game_name}_")
            if path.suffix.lower() in FIGURE_SUFFIXES and matches_game:
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

    @property
    def custom_equilibrium_figure_dir(self) -> Path:
        return self.game_catalog.custom_game_dir / ".equilibria"

    def _clear_custom_equilibrium_figures(self, game_name: str) -> None:
        if not game_name.startswith(CUSTOM_GAME_PREFIX):
            return
        slug = game_name.removeprefix(CUSTOM_GAME_PREFIX)
        if self.custom_equilibrium_figure_dir.exists():
            for suffix in FIGURE_SUFFIXES:
                for path in self.custom_equilibrium_figure_dir.glob(f"{slug}_*{suffix}"):
                    path.unlink()

    def _equilibrium_figure_path(
        self,
        game_name: str,
        equilibrium: str,
    ) -> tuple[Path, np.ndarray | None, str]:
        if not self.supports_matrix_figures(game_name):
            raise ValueError(f"unknown game: {game_name}")
        if equilibrium not in {"ce", "cce"}:
            raise ValueError(f"unknown equilibrium concept: {equilibrium}")
        if game_name in PAYOFF_FACTORIES:
            output_path = (
                PRECOMPUTED_EQUILIBRIUM_DIR
                / equilibrium_figure_filename(game_name, equilibrium)
            )
            return output_path, None, self.game_presentations[game_name]["label"]

        payoff_tensor = self.game_catalog.load(game_name)
        digest = payoff_tensor_digest(payoff_tensor)
        slug = game_name.removeprefix(CUSTOM_GAME_PREFIX)
        output_path = self.custom_equilibrium_figure_dir / equilibrium_figure_filename(
            f"{slug}_{digest}",
            equilibrium,
        )
        return output_path, payoff_tensor, self.game_definitions[game_name].label

    def equilibrium_figure(self, game_name: str, equilibrium: str, figure_format: str = "png") -> Path:
        output_path, payoff_tensor, game_label = self._equilibrium_figure_path(
            game_name,
            equilibrium,
        )
        requested_path = figure_path(output_path, figure_format)
        if requested_path.is_file():
            return requested_path
        if payoff_tensor is None and figure_format == "png":
            raise FileNotFoundError(
                f"missing precomputed equilibrium figure: {output_path.name}"
            )
        if payoff_tensor is None:
            payoff_tensor = PAYOFF_FACTORIES[game_name]()

        from experiments.plots.plot_equilibrium_weights import (
            plot_equilibrium_profile_weights,
        )

        with self._equilibrium_figure_lock:
            if requested_path.is_file():
                return requested_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".equilibrium-weights-",
                dir=output_path.parent,
            ) as temporary_directory:
                temporary_path = Path(temporary_directory) / output_path.name
                plot_equilibrium_profile_weights(
                    payoff_tensor,
                    equilibrium,
                    temporary_path,
                    game_name=game_label,
                )
                publish_figure_pair(temporary_path, output_path, overwrite=False)
        return requested_path

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
            / f"{input_path.stem}_equilibrium_distance.png",
        )

    def _group_convergence_figure_path(
        self,
        group_id: str,
    ) -> tuple[list[Path], Path, str]:
        input_paths = self._result_group_paths(group_id)
        game_name = next(iter_result_rows(input_paths[0]))["game"]
        if not self.supports_equilibrium_distance(game_name):
            raise ValueError(f"equilibrium distance is unavailable for {game_name}")
        cache_stem = self._group_cache_stem(group_id, input_paths)
        return (
            input_paths,
            self.detail_figure_dir
            / f"{cache_stem}_replicate_mean_equilibrium_distance.png",
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
            game_name = next(iter_result_rows(input_paths[0]))["game"]
            plot_result_equilibrium_distance(
                input_paths,
                temporary_path,
                game_label=self.game_presentations[game_name]["label"],
                custom_game_dir=self.game_catalog.custom_game_dir,
            )
            with self._detail_figure_lock:
                if generation != self._detail_figure_generation:
                    raise RuntimeError("equilibrium convergence figure generation was invalidated")
                if any(not path.is_file() or path.stat().st_mtime_ns != mtime for path, mtime in input_state.items()):
                    raise RuntimeError("experiment group changed while equilibrium convergence figures were generated")
                publish_figure_pair(temporary_path, output_path)
        return output_path

    def _invalidate_detail_figures(self) -> None:
        if self._experimental_trajectory_dashboard is not None:
            self._experimental_trajectory_dashboard.invalidate()
        with self._convergence_future_lock:
            for future in self._convergence_futures.values():
                future.cancel()
            self._convergence_futures.clear()
        with self._detail_figure_lock:
            self._detail_figure_generation += 1
            if self.detail_figure_dir.exists():
                for path in self.detail_figure_dir.iterdir():
                    if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES:
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
            _clear_result_files(
                (self.raw_dir, self.adversarial_raw_dir, self.adversarial_scaling_raw_dir),
                (self.figure_dir, self.adversarial_figure_dir, self.adversarial_scaling_figure_dir),
            )

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
                records.append({**_figure_file_record(path, confidence_intervals=True), **metadata})
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
                r"regret_player_(\d+)\.(?:png|pdf)",
                remainder,
            )
            scaling_match = re.fullmatch(
                r"(expected|realized)_(external|internal|swap)_"
                r"regret_over_sqrt_t_player_(\d+)\.(?:png|pdf)",
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
        return _validate_result_figure(self.figure_dir, filename, self.figure_records())
