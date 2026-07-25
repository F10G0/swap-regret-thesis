from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import Event, Lock
from typing import Any
from uuid import uuid4

from config import BANDIT_REPLICATES, HORIZON, SEED
from experiments.games import PAYOFF_FACTORIES
from experiments.scenarios.bandit_cross_play import (
    ALGORITHMS as BANDIT_ALGORITHMS,
    run_bandit_cross_play_experiment,
)
from experiments.scenarios.full_information_cross_play import (
    ALGORITHMS as FULL_INFORMATION_ALGORITHMS,
    run_full_information_cross_play_experiment,
)
from experiments.scenarios.cross_play import AlgorithmFactory
from experiments.runner import ExperimentCancelled
from experiments.spec import ExperimentSpec
from web.results import ResultIndex, ResultSnapshot
from web.validation import ExperimentForm, validate_leaf_filename


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackMode:
    label: str
    algorithms: dict[str, AlgorithmFactory]
    runner: Callable[..., Path]


FEEDBACK_MODES = {
    "full_information": FeedbackMode(
        label="Full information",
        algorithms=FULL_INFORMATION_ALGORITHMS,
        runner=run_full_information_cross_play_experiment,
    ),
    "bandit": FeedbackMode(
        label="Bandit feedback",
        algorithms=BANDIT_ALGORITHMS,
        runner=run_bandit_cross_play_experiment,
    ),
}

ALGORITHM_LABELS = {
    "hedge": "Hedge",
    "exp3": "EXP3",
    "exp3_ix": "EXP3-IX",
    "bm": "BM",
    "ito": "Ito",
    "lce_ix": "LCE-IX",
    "regret_matching": "Regret Matching",
    "stationary_regret_matching": "Stationary Regret Matching",
}


class ServiceBusyError(RuntimeError):
    pass


class PlotUpdateError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    description: str
    status: str
    message: str
    created_at: str
    reload_page: bool = True
    completed: int = 0
    total: int = 1
    cancel_requested: bool = False
    started_at: str | None = None
    finished_at: str | None = None

    def public_data(self) -> dict:
        return asdict(self)


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str):
        self.manager = manager
        self.job_id = job_id

    @property
    def cancelled(self) -> bool:
        return self.manager._cancel_requested(self.job_id)

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise ExperimentCancelled("experiment cancelled")

    def advance(self, message: str | None = None) -> None:
        self.manager._advance(self.job_id, message)


class JobManager:
    def __init__(self, max_history: int = 20):
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="swap-regret-web",
        )
        self._lock = Lock()
        self._jobs: dict[str, Job] = {}
        self._cancel_events: dict[str, Event] = {}
        self._maintenance_active = False
        self._max_history = max_history

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _has_active_job_unlocked(self) -> bool:
        return any(job.status in {"queued", "running"} for job in self._jobs.values())

    def submit(self, description: str, operation: Callable[[JobContext], str | None], reload_page: bool = True, total: int = 1) -> Job:
        with self._lock:
            if self._maintenance_active or self._has_active_job_unlocked():
                raise ServiceBusyError("another dashboard operation is already running")

            self._trim_history_unlocked()
            job = Job(
                id=uuid4().hex,
                description=description,
                status="queued",
                message="Waiting to start",
                created_at=self._now(),
                reload_page=reload_page,
                total=total,
            )
            self._jobs[job.id] = job
            self._cancel_events[job.id] = Event()
            self._executor.submit(self._run, job.id, operation)
            return Job(**asdict(job))

    def _run(self, job_id: str, operation: Callable[[JobContext], str | None]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                self._finish_cancelled_unlocked(job)
                return
            job.status = "running"
            job.message = f"0 / {job.total} completed"
            job.started_at = self._now()

        try:
            message = operation(JobContext(self, job_id))
        except ExperimentCancelled:
            with self._lock:
                self._finish_cancelled_unlocked(self._jobs[job_id])
        except Exception as error:
            logger.exception("Dashboard job %s failed", job_id)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.message = f"{type(error).__name__}: {error}"
                job.finished_at = self._now()
        else:
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    self._finish_cancelled_unlocked(job)
                else:
                    job.status = "succeeded"
                    job.completed = job.total
                    job.message = message or "Operation completed"
                    job.finished_at = self._now()

    def _finish_cancelled_unlocked(self, job: Job) -> None:
        job.status = "cancelled"
        job.message = f"Cancelled after {job.completed} / {job.total}"
        job.finished_at = self._now()

    def _cancel_requested(self, job_id: str) -> bool:
        return self._cancel_events[job_id].is_set()

    def _advance(self, job_id: str, message: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.completed = min(job.completed + 1, job.total)
            job.message = message or f"{job.completed} / {job.total} completed"

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"queued", "running"}:
                raise ValueError("job has already finished")
            job.cancel_requested = True
            self._cancel_events[job_id].set()
            job.message = "Cancellation requested"
            return Job(**asdict(job))

    def run_maintenance(self, operation: Callable[[], Any]) -> Any:
        with self._lock:
            if self._maintenance_active or self._has_active_job_unlocked():
                raise ServiceBusyError("wait for the active operation to finish")
            self._maintenance_active = True

        try:
            return operation()
        finally:
            with self._lock:
                self._maintenance_active = False

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else Job(**asdict(job))

    def recent(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
            return [Job(**asdict(job)) for job in reversed(jobs)]

    def is_busy(self) -> bool:
        with self._lock:
            return self._maintenance_active or self._has_active_job_unlocked()

    def _trim_history_unlocked(self) -> None:
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"succeeded", "failed", "cancelled"}
        ]
        excess = max(0, len(self._jobs) - self._max_history + 1)
        for job_id in terminal_ids[:excess]:
            del self._jobs[job_id]
            del self._cancel_events[job_id]


class DashboardService:
    def __init__(
        self,
        results_dir: str | Path,
        raw_dir: str | Path,
        figure_dir: str | Path,
        job_manager: JobManager | None = None,
    ):
        self.results_dir = Path(results_dir)
        self.raw_dir = Path(raw_dir)
        self.figure_dir = Path(figure_dir)
        self.jobs = job_manager or JobManager()
        self.result_index = ResultIndex(self.raw_dir)
        self._detail_figure_lock = Lock()

    @property
    def games(self) -> list[str]:
        return list(PAYOFF_FACTORIES)

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
        return {name: ALGORITHM_LABELS.get(name, name) for algorithms in self.algorithms_by_feedback_mode.values() for name in algorithms}

    def default_form_state(self) -> dict:
        feedback_mode = "full_information"
        first_algorithm = self.algorithms_by_feedback_mode[feedback_mode][0]
        return {
            "game": self.games[0],
            "feedback_mode": feedback_mode,
            "algorithm_player_0": first_algorithm,
            "algorithm_player_1": first_algorithm,
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
        )

    def _replicates(self, form: ExperimentForm) -> range:
        return range(form.replicate, form.replicate + form.replicates)

    def _missing_runs(self, form: ExperimentForm, profiles: list[list[str]]) -> tuple[list[tuple[list[str], int]], int]:
        requested = [(profile, replicate) for profile in profiles for replicate in self._replicates(form)]
        missing = [(profile, replicate) for profile, replicate in requested if not (self.raw_dir / f"{self._spec(form, profile, replicate).run_id}.csv").exists()]
        return missing, len(requested) - len(missing)

    def _run_experiments(self, form: ExperimentForm, mode: FeedbackMode, missing_runs: list[tuple[list[str], int]], skipped_count: int, job: JobContext) -> str:
        for algorithm_names, replicate in missing_runs:
            job.check_cancelled()
            mode.runner(
                game_name=form.game, algorithm_names=algorithm_names, horizon=form.horizon, seed=form.seed, replicate=replicate,
                output_dir=self.raw_dir, should_cancel=lambda: job.cancelled,
            )
            job.advance()
        job.check_cancelled()
        try:
            self._publish_plots(form.game)
        except Exception as error:
            raise PlotUpdateError(f"experiments were saved, but their figures could not be rebuilt: {error}") from error
        return f"Completed {len(missing_runs)} run(s); skipped {skipped_count} existing"

    def submit_experiment(self, form: ExperimentForm) -> Job:
        mode = FEEDBACK_MODES[form.feedback_mode]
        missing_runs, skipped_count = self._missing_runs(form, [form.algorithm_names])
        if not missing_runs:
            raise FileExistsError("all requested replicates already exist")

        def operation(job: JobContext) -> str:
            return self._run_experiments(form, mode, missing_runs, skipped_count, job)

        return self.jobs.submit(
            f"{form.game}: {' vs '.join(ALGORITHM_LABELS.get(name, name) for name in form.algorithm_names)}",
            operation,
            total=len(missing_runs),
        )

    def submit_all_pairs(self, form: ExperimentForm) -> tuple[Job, int, int]:
        mode = FEEDBACK_MODES[form.feedback_mode]
        pairs = [
            [algorithm_player_0, algorithm_player_1]
            for algorithm_player_0 in mode.algorithms
            for algorithm_player_1 in mode.algorithms
        ]
        missing_runs, skipped_count = self._missing_runs(form, pairs)
        if not missing_runs:
            raise FileExistsError("all requested runs already exist")

        def operation(job: JobContext) -> str:
            return self._run_experiments(form, mode, missing_runs, skipped_count, job)

        job = self.jobs.submit(
            f"{form.game}: all {form.feedback_mode} pairs",
            operation,
            total=len(missing_runs),
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

    def joint_action_figure(self, filename: str) -> Path:
        filename = validate_leaf_filename(filename, ".csv")
        input_path = self.raw_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(filename)

        output_path = self.detail_figure_dir / f"{input_path.stem}_joint_actions.png"
        with self._detail_figure_lock:
            if output_path.is_file() and output_path.stat().st_mtime_ns >= input_path.stat().st_mtime_ns:
                return output_path

            from experiments.plots.plot_joint_actions import plot_joint_actions

            self.detail_figure_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".joint-actions-", dir=self.detail_figure_dir) as temporary_directory:
                temporary_path = Path(temporary_directory) / output_path.name
                plot_joint_actions(input_path, temporary_path)
                os.replace(temporary_path, output_path)
        return output_path

    def delete_experiment(self, filename: str) -> None:
        filename = validate_leaf_filename(filename, ".csv")

        def operation() -> None:
            csv_path = self.raw_dir / filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"experiment {filename} does not exist")

            csv_path.unlink()
            (self.detail_figure_dir / f"{csv_path.stem}_joint_actions.png").unlink(missing_ok=True)
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
            if self.raw_dir.exists():
                for path in self.raw_dir.glob("*.csv"):
                    path.unlink()
            if self.figure_dir.exists():
                for path in self.figure_dir.rglob("*.png"):
                    path.unlink()

            report_path = self.results_dir / "index.html"
            if report_path.is_file():
                report_path.unlink()

        self.jobs.run_maintenance(operation)

    def experiment_filenames(self) -> list[str]:
        return self.result_snapshot().filenames

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

    def experiment_summaries(self) -> tuple[list[dict], list[str]]:
        snapshot = self.result_snapshot()
        return snapshot.summaries, snapshot.warnings

    def validate_csv_filename(self, filename: str) -> str:
        return validate_leaf_filename(filename, ".csv")

    def validate_figure_filename(self, filename: str) -> str:
        filename = validate_leaf_filename(filename, ".png")
        if self._parse_figure_filename(filename) is None:
            raise ValueError("unknown figure filename")
        return filename
