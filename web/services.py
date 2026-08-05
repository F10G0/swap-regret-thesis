from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

import numpy as np

from config import (
    ADVERSARIAL_ACTIONS,
    ADVERSARIAL_MEMORY_WINDOW,
    BANDIT_REPLICATES,
    CUSTOM_GAME_DIR,
    HORIZON,
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
    figure_pair_is_current,
    figure_path,
    figure_paths,
    publish_figure_pair,
)
from experiments.results import iter_result_rows
from experiments.spec import ExperimentSpec
from experiments.scenarios.adversarial import (
    ALGORITHMS as ADVERSARIAL_ALGORITHMS,
    AdversarialExperimentSpec,
    TARGET_REGRET_BY_ALGORITHM,
    adversarial_memory_label,
    adversarial_memory_window,
    load_final_adversarial_row,
    run_adversarial_experiment,
)
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.experiment_modes import FEEDBACK_MODES, FeedbackMode
from web.jobs import Job, JobContext, JobManager
from web.presentations import GAME_PRESENTATIONS
from web.result_groups import (
    result_group_filenames,
)
from web.result_index import ResultIndex, ResultSnapshot
from web.validation import (
    AdversarialExperimentForm,
    ExperimentForm,
    validate_leaf_filename,
)


logger = logging.getLogger(__name__)
EQUILIBRIUM_DISTANCE_FIGURE = "distance"


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
    def adversarial_algorithms(self) -> list[str]:
        return list(ADVERSARIAL_ALGORITHMS)

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
            "replicates": BANDIT_REPLICATES,
        }

    def default_adversarial_form_state(self) -> dict:
        return {
            "algorithm_name": self.adversarial_algorithms[0],
            "n_actions": ADVERSARIAL_ACTIONS,
            "memory_window": ADVERSARIAL_MEMORY_WINDOW,
            "horizon": HORIZON,
            "seed": SEED,
        }

    def submit_adversarial_experiment(
        self,
        form: AdversarialExperimentForm,
    ) -> Job:
        spec = AdversarialExperimentSpec(
            algorithm_name=form.algorithm_name,
            n_actions=form.n_actions,
            memory_window=form.memory_window,
            horizon=form.horizon,
            seed=form.seed,
        )
        resource_key = f"adversarial:{spec.run_id}"
        reserved = self.jobs.reserved_resources()
        if (
            resource_key in reserved
            or (self.adversarial_raw_dir / f"{spec.run_id}.csv").exists()
        ):
            raise FileExistsError(
                "the requested adversarial run already exists or is queued"
            )

        def operation(job: JobContext) -> str:
            job.check_cancelled()
            run_adversarial_experiment(
                algorithm_name=spec.algorithm_name,
                n_actions=spec.n_actions,
                memory_window=spec.memory_window,
                horizon=spec.horizon,
                seed=spec.seed,
                output_dir=self.adversarial_raw_dir,
                should_cancel=lambda: job.cancelled,
            )
            job.advance()
            job.check_cancelled()
            try:
                self._publish_adversarial_plots()
            except Exception as error:
                raise PlotUpdateError(
                    "adversarial runs were saved, but their figures could not "
                    f"be rebuilt: {error}"
                ) from error
            return "Completed adversarial run"

        return self.jobs.submit(
            (
                f"Adversarial: {algorithm_label(form.algorithm_name)} · "
                f"{form.n_actions} actions · "
                f"{adversarial_memory_label(spec.memory_window)}"
            ),
            operation,
            resource_keys={resource_key},
        )

    def _publish_adversarial_plots(self) -> None:
        from experiments.plots.plot_adversarial import plot_adversarial_results

        self.adversarial_dir.mkdir(parents=True, exist_ok=True)
        self.adversarial_figure_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".adversarial-figures-",
            dir=self.adversarial_dir,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            generated = plot_adversarial_results(
                self.adversarial_raw_dir,
                temporary_path,
                skip_invalid=True,
            )
            generated_paths = [
                companion
                for path in generated
                for companion in figure_paths(path)
            ]
            generated_names = {path.name for path in generated_paths}
            for generated_path in generated_paths:
                os.replace(generated_path, self.adversarial_figure_dir / generated_path.name)
            for old_path in self.adversarial_figure_dir.iterdir():
                if old_path.suffix.lower() not in FIGURE_SUFFIXES:
                    continue
                if old_path.name not in generated_names:
                    old_path.unlink()

    def adversarial_result_summaries(self) -> tuple[list[dict], list[str]]:
        summaries = []
        warnings = []
        for path in sorted(self.adversarial_raw_dir.glob("*.csv")):
            try:
                row = load_final_adversarial_row(path)
                algorithm = row["algorithm"]
                target_regret = TARGET_REGRET_BY_ALGORITHM.get(
                    algorithm,
                    "external",
                )
                memory_window = adversarial_memory_window(row["memory"])
                summaries.append(
                    {
                        "filename": path.name,
                        "algorithm": algorithm,
                        "algorithm_label": algorithm_label(algorithm),
                        "n_actions": int(row["n_actions"]),
                        "memory_label": adversarial_memory_label(memory_window),
                        "horizon": int(row["horizon"]),
                        "seed": int(row["seed"]),
                        "target_regret": target_regret,
                        "expected_regret": float(
                            row[
                                f"average_expected_{target_regret}_regret"
                            ]
                        ),
                        "realized_regret": float(
                            row[
                                f"average_realized_{target_regret}_regret"
                            ]
                        ),
                    }
                )
            except (OSError, TypeError, ValueError) as error:
                warnings.append(f"Skipped {path.name}: {error}")
        return summaries, warnings

    def adversarial_figure_records(self) -> list[dict]:
        records = []
        regret_pattern = re.compile(
            r"historical_frequency_(\d+)_actions_(average_)?"
            r"(expected|realized)_(external|internal|swap)_regret"
            r"(_over_sqrt_t)?\.png"
        )
        run_metadata = {}
        for path in sorted(self.adversarial_figure_dir.glob("*.png")):
            match = regret_pattern.fullmatch(path.name)
            if match is not None:
                average = match.group(2) is not None
                scaled = match.group(5) is not None
                if average == scaled:
                    continue
                pdf_path = path.with_suffix(".pdf")
                records.append(
                    {
                        "filename": path.name,
                        "pdf_filename": pdf_path.name if pdf_path.is_file() else None,
                        "kind": "regret",
                        "n_actions": int(match.group(1)),
                        "source": match.group(3),
                        "regret": match.group(4),
                        "view": "average" if average else "sqrt_scaling",
                    }
                )
                continue

            diagnostic = next(
                (
                    name
                    for name in ("punished_action", "action")
                    if path.name.endswith(f"_{name}_frequency.png")
                ),
                None,
            )
            if diagnostic is None:
                continue
            run_stem = path.name.removesuffix(
                f"_{diagnostic}_frequency.png"
            )
            try:
                if run_stem not in run_metadata:
                    run_metadata[run_stem] = load_final_adversarial_row(
                        self.adversarial_raw_dir / f"{run_stem}.csv"
                    )
                row = run_metadata[run_stem]
                memory_window = adversarial_memory_window(row["memory"])
                pdf_path = path.with_suffix(".pdf")
                records.append(
                    {
                        "filename": path.name,
                        "pdf_filename": pdf_path.name if pdf_path.is_file() else None,
                        "kind": "behavior",
                        "diagnostic": diagnostic,
                        "diagnostic_label": (
                            "Learner action frequency"
                            if diagnostic == "action"
                            else "Punished-action frequency"
                        ),
                        "algorithm_label": algorithm_label(row["algorithm"]),
                        "memory_label": adversarial_memory_label(memory_window),
                        "n_actions": int(row["n_actions"]),
                    }
                )
            except (OSError, TypeError, ValueError, KeyError):
                continue
        regret_order = {"external": 0, "internal": 1, "swap": 2}
        view_order = {"average": 0, "sqrt_scaling": 1}
        return sorted(
            records,
            key=lambda record: (
                0 if record["kind"] == "regret" else 1,
                record.get("source", ""),
                view_order.get(record.get("view", ""), 0),
                record["n_actions"],
                regret_order.get(record.get("regret", ""), 0),
                record["filename"],
            ),
        )

    def validate_adversarial_csv_filename(self, filename: str) -> str:
        filename = validate_leaf_filename(filename, ".csv")
        if not (self.adversarial_raw_dir / filename).is_file():
            raise FileNotFoundError(filename)
        return filename

    def validate_adversarial_figure_filename(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in FIGURE_SUFFIXES:
            raise ValueError("invalid figure filename")
        filename = validate_leaf_filename(filename, suffix)
        preview_name = Path(filename).with_suffix(".png").name
        if not any(
            record["filename"] == preview_name
            for record in self.adversarial_figure_records()
        ) or not (self.adversarial_figure_dir / filename).is_file():
            raise FileNotFoundError(filename)
        return filename

    def delete_adversarial_experiment(self, filename: str) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            path = self.adversarial_raw_dir / filename
            if not path.is_file():
                raise FileNotFoundError(filename)
            path.unlink()
            self._publish_adversarial_plots()

        self.jobs.run_maintenance(operation)

    def clear_adversarial_results(self) -> tuple[int, int]:
        def operation() -> tuple[int, int]:
            csv_paths = list(self.adversarial_raw_dir.glob("*.csv"))
            figures = [path for path in self.adversarial_figure_dir.glob("*") if path.suffix.lower() in FIGURE_SUFFIXES]
            figure_count = sum(path.suffix.lower() == ".png" for path in figures)
            paths = csv_paths + figures
            for path in paths:
                path.unlink()
            return len(csv_paths), figure_count

        return self.jobs.run_maintenance(operation)

    def _spec(self, form: ExperimentForm, algorithm_names: list[str] | None = None, replicate: int | None = None) -> ExperimentSpec:
        names = algorithm_names or form.algorithm_names
        return ExperimentSpec(
            game_name=form.game,
            feedback_mode=form.feedback_mode,
            algorithm_names=tuple(names),
            horizon=form.horizon,
            seed=form.seed,
            replicate=0 if replicate is None else replicate,
            regret_evaluation=form.regret_evaluation,
            game_payoff_digest=payoff_tensor_digest(self.game_catalog.load(form.game)),
        )

    def _replicates(self, form: ExperimentForm) -> range:
        if form.feedback_mode != "bandit":
            return range(1)
        return range(form.replicates)

    def _missing_runs(self, form: ExperimentForm) -> tuple[list[tuple[list[str], int]], int]:
        profile = list(form.algorithm_names)
        requested = [(profile, replicate) for replicate in self._replicates(form)]
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
        missing_runs, skipped_count = self._missing_runs(form)
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

    def submit_plot_rebuild(self) -> Job:
        return self.jobs.submit(
            "Rebuild all figures",
            lambda job: self._rebuild_all_plots(job),
            reload_page=False,
        )

    def _rebuild_all_plots(self, job: JobContext) -> str:
        job.check_cancelled()
        self._publish_plots()
        self._publish_adversarial_plots()
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
                matches_target = lambda path: path.suffix.lower() in FIGURE_SUFFIXES
            else:
                plot_selected_results(game_name, self.raw_dir, temporary_path, skip_invalid=True)
                matches_target = lambda path: (
                    path.suffix.lower() in FIGURE_SUFFIXES
                    and path.name.startswith(f"{game_name}_")
                )

            generated_paths = [
                path
                for path in temporary_path.iterdir()
                if path.suffix.lower() in FIGURE_SUFFIXES
            ]
            generated_names = {path.name for path in generated_paths}
            for generated_path in generated_paths:
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

    @staticmethod
    def _validate_convergence_figure(figure: str) -> None:
        if figure != EQUILIBRIUM_DISTANCE_FIGURE:
            raise ValueError(
                f"unknown equilibrium convergence figure: {figure}"
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
        figure: str,
    ) -> tuple[Path | None, str | None]:
        self._validate_convergence_figure(figure)
        input_path, output_path = self._convergence_figure_path(filename)
        return self._request_convergence_figure(
            [input_path],
            output_path,
            f"{filename}:{figure}",
            lambda: self._generate_equilibrium_distance(
                [input_path],
                output_path,
            ),
            filename,
        )

    def request_group_equilibrium_convergence_figure(
        self,
        group_id: str,
        figure: str,
    ) -> tuple[Path | None, str | None]:
        self._validate_convergence_figure(figure)
        input_paths, output_path, cache_stem = self._group_convergence_figure_path(
            group_id
        )
        return self._request_convergence_figure(
            input_paths,
            output_path,
            f"group:{cache_stem}:{figure}",
            lambda: self._generate_equilibrium_distance(
                input_paths,
                output_path,
            ),
            f"group {group_id}",
        )

    def equilibrium_convergence_figures(
        self,
        filename: str,
    ) -> dict[str, Path]:
        input_path, output_path = self._convergence_figure_path(filename)
        return {
            EQUILIBRIUM_DISTANCE_FIGURE: (
                self._generate_equilibrium_distance(
                    [input_path],
                    output_path,
                )
            )
        }

    def group_equilibrium_convergence_figures(
        self,
        group_id: str,
    ) -> dict[str, Path]:
        input_paths, output_path, _ = self._group_convergence_figure_path(
            group_id
        )
        return {
            EQUILIBRIUM_DISTANCE_FIGURE: (
                self._generate_equilibrium_distance(
                    input_paths,
                    output_path,
                )
            )
        }

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
            if self.raw_dir.exists():
                for path in self.raw_dir.glob("*.csv"):
                    path.unlink()
            if self.figure_dir.exists():
                for path in self.figure_dir.iterdir():
                    if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES:
                        path.unlink()
            if self.adversarial_raw_dir.exists():
                for path in self.adversarial_raw_dir.glob("*.csv"):
                    path.unlink()
            if self.adversarial_figure_dir.exists():
                for path in self.adversarial_figure_dir.iterdir():
                    if path.suffix.lower() in FIGURE_SUFFIXES:
                        path.unlink()

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
                pdf_path = path.with_suffix(".pdf")
                records.append({"filename": path.name, "pdf_filename": pdf_path.name if pdf_path.is_file() else None, **metadata})
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
        suffix = Path(filename).suffix.lower()
        if suffix not in FIGURE_SUFFIXES:
            raise ValueError("invalid figure filename")
        filename = validate_leaf_filename(filename, suffix)
        if self._parse_figure_filename(filename) is None or not (self.figure_dir / filename).is_file():
            raise ValueError("unknown figure filename")
        return filename
