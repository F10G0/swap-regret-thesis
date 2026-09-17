"""Bounded, spawn-based execution of independent experiment replicates."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os

from experiments.runner import ExperimentCancelled


MIN_PARALLEL_ROUNDS = 20_000
MAX_PARALLEL_WORKERS = 12
_worker_cancel = None
_worker_progress = None


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    return max(1, cpu_count() or 1)


def _replicate_worker_count(tasks: list[dict], workers: int | None) -> int:
    available = _available_cpu_count()
    count = min(workers or available, available, len(tasks), MAX_PARALLEL_WORKERS)
    if _worker_cancel is not None or multiprocessing.current_process().name != "MainProcess":
        return 1  # Never nest pools inside a replicate worker.
    if workers is None and sum(task.get("horizon", 0) for task in tasks) < MIN_PARALLEL_ROUNDS:
        return 1
    return count


def _initialize_worker(cancel_event, progress_counter) -> None:
    global _worker_cancel, _worker_progress
    _worker_cancel = cancel_event
    _worker_progress = progress_counter


def _report_rounds(delta_rounds: int) -> None:
    with _worker_progress.get_lock():
        _worker_progress.value += delta_rounds


def _execute(function, kwargs):
    if _worker_cancel.is_set():
        raise ExperimentCancelled("experiment cancelled")
    if _worker_progress is not None:
        return function(**kwargs, should_cancel=_worker_cancel.is_set, report_rounds=_report_rounds)
    return function(**kwargs, should_cancel=_worker_cancel.is_set)


def run_replicates(function, tasks: list[dict], *, workers: int | None = None,
                   should_cancel=None, completed=None, round_progress=None) -> list:
    """Run keyword-argument tasks; return results in input order, not finish order.

    Only top-level callables and serializable task arguments enter workers. UI
    callbacks stay in the parent. Small batches run serially by default; an
    explicit workers=1/2/... selects the execution mode for reproducibility tests.
    Automatic and explicit worker counts are capped by available CPUs, task
    count, and MAX_PARALLEL_WORKERS.
    Numerical-library settings are inherited unchanged in both modes.
    """
    if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0):
        raise ValueError("workers must be a positive integer")
    if not tasks:
        return []
    count = _replicate_worker_count(tasks, workers)

    def check_cancelled():
        if should_cancel is not None and should_cancel():
            raise ExperimentCancelled("experiment cancelled")

    if count == 1:
        results = []
        rounds_completed = 0

        def report_rounds(delta_rounds):
            nonlocal rounds_completed
            rounds_completed += delta_rounds
            round_progress(rounds_completed)

        for task in tasks:
            check_cancelled()
            progress_kwargs = {"report_rounds": report_rounds} if round_progress is not None else {}
            results.append(function(**task, should_cancel=should_cancel, **progress_kwargs))
            if completed is not None:
                completed()
        check_cancelled()
        return results

    check_cancelled()
    context = multiprocessing.get_context("spawn")
    cancel_event = context.Event()
    progress_counter = context.Value("q", 0) if round_progress is not None else None
    results = [None] * len(tasks)
    reported_rounds = 0

    def sample_round_progress():
        nonlocal reported_rounds
        current = progress_counter.value if progress_counter is not None else reported_rounds
        if current != reported_rounds:
            reported_rounds = current
            round_progress(reported_rounds)

    # No unbounded submission queue: at most one in-flight task per worker.
    with ProcessPoolExecutor(max_workers=count, mp_context=context,
                             initializer=_initialize_worker, initargs=(cancel_event, progress_counter)) as pool:
        pending = {}
        next_index = 0
        try:
            while next_index < len(tasks) or pending:
                check_cancelled()
                while next_index < len(tasks) and len(pending) < count:
                    future = pool.submit(_execute, function, tasks[next_index])
                    pending[future] = next_index
                    next_index += 1
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                sample_round_progress()
                for future in done:
                    index = pending.pop(future)
                    results[index] = future.result()
                    if completed is not None:
                        completed()
            check_cancelled()
            sample_round_progress()
        except BaseException:
            sample_round_progress()
            cancel_event.set()
            for future in pending:
                future.cancel()
            raise
    return results
