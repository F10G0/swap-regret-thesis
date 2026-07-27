from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from threading import Event, Lock
from typing import Any
from uuid import uuid4

from experiments.runner import ExperimentCancelled


logger = logging.getLogger(__name__)


class ServiceBusyError(RuntimeError):
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swap-regret-web")
        self._lock = Lock()
        self._jobs: dict[str, Job] = {}
        self._cancel_events: dict[str, Event] = {}
        self._job_resource_keys: dict[str, set[str]] = {}
        self._resource_owners: dict[str, str] = {}
        self._maintenance_active = False
        self._max_history = max_history

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _has_active_job_unlocked(self) -> bool:
        return any(job.status in {"queued", "running"} for job in self._jobs.values())

    def submit(
        self,
        description: str,
        operation: Callable[[JobContext], str | None],
        reload_page: bool = True,
        total: int = 1,
        resource_keys: set[str] | None = None,
    ) -> Job:
        with self._lock:
            if self._maintenance_active:
                raise ServiceBusyError("dashboard maintenance is currently running")

            self._trim_history_unlocked()
            resource_keys = set(resource_keys or ())
            if resource_keys & self._resource_owners.keys():
                raise FileExistsError("one or more requested runs are already queued")
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
            self._job_resource_keys[job.id] = resource_keys
            for resource_key in resource_keys:
                self._resource_owners[resource_key] = job.id
            self._executor.submit(self._run, job.id, operation)
            return Job(**asdict(job))

    def _run(self, job_id: str, operation: Callable[[JobContext], str | None]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_requested:
                self._finish_cancelled_unlocked(job)
                self._release_resources_unlocked(job_id)
                return
            job.status = "running"
            job.message = f"0 / {job.total} completed"
            job.started_at = self._now()

        try:
            message = operation(JobContext(self, job_id))
        except ExperimentCancelled:
            with self._lock:
                self._finish_cancelled_unlocked(self._jobs[job_id])
                self._release_resources_unlocked(job_id)
        except Exception as error:
            logger.exception("Dashboard job %s failed", job_id)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.message = f"{type(error).__name__}: {error}"
                job.finished_at = self._now()
                self._release_resources_unlocked(job_id)
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
                self._release_resources_unlocked(job_id)

    def _finish_cancelled_unlocked(self, job: Job) -> None:
        job.status = "cancelled"
        job.message = f"Cancelled after {job.completed} / {job.total}"
        job.finished_at = self._now()

    def _cancel_requested(self, job_id: str) -> bool:
        return self._cancel_events[job_id].is_set()

    def _release_resources_unlocked(self, job_id: str) -> None:
        for resource_key in self._job_resource_keys.pop(job_id, set()):
            if self._resource_owners.get(resource_key) == job_id:
                del self._resource_owners[resource_key]

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
            if job.status == "queued":
                self._finish_cancelled_unlocked(job)
                self._release_resources_unlocked(job_id)
            else:
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
            return [Job(**asdict(job)) for job in reversed(self._jobs.values())]

    def reserved_resources(self) -> set[str]:
        with self._lock:
            return set(self._resource_owners)

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
