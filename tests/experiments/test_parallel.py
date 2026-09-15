from functools import partial
import os
from pathlib import Path
import time

import pytest

import experiments.parallel as parallel
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.parallel import run_replicates
from experiments.recorder import CsvRecorder
from experiments.runner import ExperimentCancelled
from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from experiments.scenarios.adversarial_scaling import AdversarialScalingSpec, run_adversarial_scaling_experiment


@pytest.mark.parametrize("runner,options", [
    (partial(run_cross_play_experiment, feedback_mode="bandit"), dict(game_name="rps", algorithm_names=["auer_exp3", "bm"])),
    (partial(run_cross_play_experiment, feedback_mode="full_information"), dict(game_name="rps", algorithm_names=["hedge", "ito"])),
    (run_adversarial_experiment, dict(algorithm_name="auer_exp3", feedback_mode="bandit", environment=RANDOM_WALK_ENVIRONMENT)),
    *[(run_adversarial_experiment, dict(algorithm_name=name, feedback_mode="bandit", n_actions=9))
      for name in ("exp3_ix", "bm", "lce_ix", "ito")],
    (partial(run_cross_play_experiment, feedback_mode="bandit"), dict(game_name="rps", algorithm_names=["ito", "bm"])),
])
def test_serial_and_process_replicates_produce_identical_csv_bytes(tmp_path, runner, options):
    results = []
    for workers in (1, 2):
        completions = []
        tasks = [dict(**options, horizon=31, seed=7, replicate=r,
                      max_recorded_points=6, output_dir=tmp_path / str(workers)) for r in (2, 0, 1)]
        paths = run_replicates(runner, tasks, workers=workers, completed=lambda: completions.append(True))
        assert len(completions) == 3
        results.append([(path.name, path.read_bytes()) for path in paths])
    assert results[0] == results[1]


def test_scaling_parallel_output_order_and_identity_are_deterministic(tmp_path):
    spec = AdversarialScalingSpec(
        environment=RANDOM_WALK_ENVIRONMENT, feedback_mode="bandit",
        algorithm_name="auer_exp3", action_counts=(2, 4), replicates=2, horizon=25,
        seed=7,
    )
    serial = run_adversarial_scaling_experiment(spec, tmp_path / "serial", workers=1)
    parallel = run_adversarial_scaling_experiment(spec, tmp_path / "parallel", workers=2)
    assert serial.name == parallel.name
    assert serial.read_bytes() == parallel.read_bytes()


def identify_worker(value, horizon=0, should_cancel=None):
    return value, os.getpid()


def nested_worker(value, should_cancel=None):
    parent = os.getpid()
    children = run_replicates(identify_worker, [dict(value=1), dict(value=2)], workers=2)
    return parent, children


def test_parallel_uses_processes_and_never_nests_pools():
    results = run_replicates(nested_worker, [dict(value=1), dict(value=2)], workers=2)
    assert all(parent != os.getpid() for parent, _ in results)
    assert all(pid == parent for parent, children in results for _, pid in children)


def test_automatic_execution_parallelizes_large_batches_but_not_small_ones():
    small = run_replicates(identify_worker, [dict(value=i, horizon=10) for i in range(2)])
    large = run_replicates(identify_worker, [dict(value=i, horizon=1_000_000) for i in range(2)])
    assert all(pid == os.getpid() for _, pid in small)
    assert all(pid != os.getpid() for _, pid in large)


def cancellable_worker(output_dir, value, should_cancel=None):
    with CsvRecorder(["value"], Path(output_dir) / f"{value}.csv") as recorder:
        # Marker lets the parent cancel only after a worker has started writing.
        (Path(output_dir) / f"{value}.started").touch()
        for _ in range(1000):
            if should_cancel():
                raise ExperimentCancelled("experiment cancelled")
            time.sleep(0.005)
        recorder.record(dict(value=value))


def test_parallel_cancellation_cleans_unfinished_csvs(tmp_path):
    with pytest.raises(ExperimentCancelled):
        run_replicates(cancellable_worker,
                       [dict(output_dir=tmp_path, value=i) for i in range(4)], workers=2,
                       should_cancel=lambda: any(tmp_path.glob("*.started")))
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("*.started"))) <= 2


def failing_worker(output_dir, value, should_cancel=None):
    if value == 0:
        raise ValueError("worker failed")
    return cancellable_worker(output_dir, value, should_cancel)


def test_worker_failure_cancels_peers_and_cleans_temporary_results(tmp_path):
    with pytest.raises(ValueError, match="worker failed"):
        run_replicates(failing_worker, [dict(output_dir=tmp_path, value=i) for i in range(4)], workers=2)
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_worker_count_is_rejected(workers):
    with pytest.raises(ValueError, match="workers"):
        run_replicates(identify_worker, [dict(value=1)], workers=workers)


@pytest.mark.parametrize("cpus,tasks,workers,expected", [
    (16, 30, None, 12), (8, 30, None, 8), (16, 5, None, 5),
    (16, 30, 16, 12), (16, 30, 4, 4),
    (16, 30, 1, 1), (1, 10, None, 1),
])
def test_worker_count_is_bounded_by_cap_cpus_and_tasks(monkeypatch, cpus, tasks, workers, expected):
    monkeypatch.setattr(parallel, "_available_cpu_count", lambda: cpus)
    assert parallel._replicate_worker_count([dict(horizon=10_000)] * tasks, workers) == expected


def test_cpu_detection_honors_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2, 3, 4, 5, 6, 7}, raising=False)
    assert parallel._available_cpu_count() == 6


def test_cpu_detection_falls_back_when_affinity_is_unavailable(monkeypatch):
    def unavailable(pid):
        raise OSError("affinity is unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 12, raising=False)
    assert parallel._available_cpu_count() == 12
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)
    assert parallel._available_cpu_count() == 1


def test_six_worker_execution_preserves_serial_csv_bytes(tmp_path, monkeypatch):
    if parallel._available_cpu_count() < 6:
        pytest.skip("six available CPUs are required for this integration test")
    counts = []
    executor = parallel.ProcessPoolExecutor

    def record_executor(*args, **kwargs):
        counts.append(kwargs["max_workers"])
        return executor(*args, **kwargs)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", record_executor)
    results = []
    for workers in (1, 6):
        tasks = [dict(algorithm_name="auer_exp3", feedback_mode="bandit", horizon=23, seed=7,
                      replicate=r, output_dir=tmp_path / str(workers))
                 for r in range(6)]
        paths = run_replicates(run_adversarial_experiment, tasks, workers=workers)
        results.append([(path.name, path.read_bytes()) for path in paths])
    assert counts == [6]
    assert results[0] == results[1]
