from pathlib import Path
import logging
from threading import Event
import time

import pytest

from web.jobs import Job, JobManager, ServiceBusyError
from web.services import PlotUpdateError
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.web.support import block_job_queue, create_service, create_test_app, wait_for_async_result, wait_for_job
from web.validation import ExperimentForm


def write_figure_pair(output_path, content: bytes) -> None:
    output_path = Path(output_path)
    output_path.write_bytes(content)
    output_path.with_suffix(".pdf").write_bytes(content)


def wait_for_equilibrium_figure(request_figure):
    return wait_for_async_result(request_figure, timeout=3)


def experiment_form() -> ExperimentForm:
    return ExperimentForm(
        game="rps",
        feedback_mode="full_information",
        algorithm_names=("hedge", "hedge"),
        horizon=2,
        seed=42,
        replicates=1,
    )


def test_job_manager_runs_queued_operations_in_submission_order() -> None:
    manager = JobManager()
    first_started = Event()
    second_started = Event()
    release_first = Event()
    execution_order = []

    def first_operation(job) -> str:
        execution_order.append("first")
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first done"

    def second_operation(job) -> str:
        execution_order.append("second")
        second_started.set()
        return "second done"

    first = manager.submit("first", first_operation, resource_keys={"first-run"})
    assert first_started.wait(timeout=1)
    second = manager.submit("second", second_operation, resource_keys={"second-run"})
    assert manager.get(second.id).status == "queued"
    assert not second_started.is_set()
    assert manager.reserved_resources() == {"first-run", "second-run"}
    release_first.set()

    assert wait_for_job(manager, first.id) == "succeeded"
    assert wait_for_job(manager, second.id) == "succeeded"

    assert execution_order == ["first", "second"]
    assert manager.reserved_resources() == set()


def test_cancelling_queued_job_releases_its_reserved_resources() -> None:
    manager = JobManager()
    first_started = Event()
    release_first = Event()

    def blocking_operation(job) -> None:
        first_started.set()
        assert release_first.wait(timeout=2)

    first = manager.submit("first", blocking_operation)
    assert first_started.wait(timeout=1)
    queued = manager.submit("queued", lambda job: None, resource_keys={"run"})

    cancelled = manager.cancel(queued.id)
    replacement = manager.submit("replacement", lambda job: None, resource_keys={"run"})

    assert cancelled.status == "cancelled"
    assert manager.get(queued.id).status == "cancelled"
    assert manager.get(replacement.id).status == "queued"
    release_first.set()
    assert wait_for_job(manager, first.id) == "succeeded"
    assert wait_for_job(manager, replacement.id) == "succeeded"


def test_maintenance_remains_blocked_while_jobs_are_queued() -> None:
    manager = JobManager()
    started = Event()
    release = Event()

    def operation(job) -> None:
        started.set()
        assert release.wait(timeout=2)

    job = manager.submit("active", operation)
    assert started.wait(timeout=1)

    with pytest.raises(ServiceBusyError, match="active operation"):
        manager.run_maintenance(lambda: None)

    release.set()
    assert wait_for_job(manager, job.id) == "succeeded"


def test_job_manager_records_and_logs_failures(caplog) -> None:
    manager = JobManager()
    caplog.set_level(logging.ERROR, logger="web.jobs")

    def operation(job) -> None:
        raise RuntimeError("failed operation")

    job = manager.submit("failure", operation)
    assert wait_for_job(manager, job.id) == "failed"
    failed_job = manager.get(job.id)

    assert failed_job.message == "RuntimeError: failed operation"
    assert "Dashboard job" in caplog.text


def test_job_manager_reports_progress_and_cancels() -> None:
    manager = JobManager()
    started = Event()

    def operation(job) -> None:
        job.advance()
        started.set()
        while not job.cancelled:
            time.sleep(0.001)
        job.check_cancelled()

    submitted = manager.submit("cancellable", operation, total=4)
    assert started.wait(timeout=1)
    manager.cancel(submitted.id)

    assert wait_for_job(manager, submitted.id) == "cancelled"
    job = manager.get(submitted.id)

    assert job.completed == 1
    assert job.total == 4


def test_clear_results_removes_derived_tree_and_preserves_inputs(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("keep me", 2, [2, 2], 0)
    input_path = service.game_catalog.custom_path(definition.id)
    artifact_locations = (
        service.raw_dir,
        service.detail_figure_dir,
        service.adversarial_dir / "cache",
        service.results_dir / "cache" / "figure_builder",
    )
    placeholders = set()
    for index, directory in enumerate(artifact_locations):
        directory.mkdir(parents=True, exist_ok=True)
        placeholder = directory / ".gitkeep"
        placeholder.write_text("", encoding="utf-8")
        placeholders.add(placeholder)
        (directory / f"derived-{index}.artifact").write_bytes(b"generated")

    service.clear_results()

    assert input_path.is_file()
    remaining_files = {path for path in tmp_path.rglob("*") if path.is_file()}
    assert remaining_files == placeholders | {input_path}


def test_custom_game_deletion_requires_its_experiments_to_be_deleted_first(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("delete me", 2, [2, 2], 0)
    service.raw_dir.mkdir(parents=True)
    result_path = service.raw_dir / f"{definition.id}_run.csv"
    result_path.write_text("result", encoding="utf-8")

    with pytest.raises(ValueError, match="recorded experiments"):
        service.delete_custom_game(definition.id)

    assert definition.id in service.game_definitions
    result_path.unlink()
    assert service.delete_custom_game(definition.id) == definition
    assert definition.id not in service.game_definitions


@pytest.mark.parametrize(
    ("payoff_player", "row_player", "column_player", "fixed_actions", "message"),
    [
        (3, 0, 1, [0, 0, 0], "payoff player"),
        (0, 1, 1, [0, 0, 0], "must be different"),
        (0, 0, 1, [0, 0], "one fixed action"),
        (0, 0, 1, [0, 0, 2], "player 2"),
    ],
)
def test_custom_game_payoff_slice_rejects_invalid_selection(
    tmp_path: Path,
    payoff_player: int,
    row_player: int,
    column_player: int,
    fixed_actions: list[int],
    message: str,
) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("inspect me", 3, [2, 3, 2], 7)

    with pytest.raises(ValueError, match=message):
        service.custom_game_payoff_slice(definition.id, payoff_player, row_player, column_player, fixed_actions)


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_ordinary_deletion_preserves_results_until_invalidation_succeeds(tmp_path, monkeypatch, kind):
    service = create_service(tmp_path)
    raw_dir = service.raw_dir if kind == "fixed" else service.adversarial_raw_dir
    delete = service.delete_experiment if kind == "fixed" else service.delete_adversarial_experiment
    raw_dir.mkdir(parents=True)
    source = raw_dir / "result.csv"
    source.write_bytes(b"recorded-result")
    retained = raw_dir / "retained.csv"
    retained.write_bytes(b"retained-result")
    artifact = service.figure_builder.output_dir / "derived.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"generated")
    with monkeypatch.context() as patch:
        patch.setattr(service, "_clear_experiment_caches", lambda: (_ for _ in ()).throw(OSError("cleanup failed")))
        with pytest.raises(OSError, match="cleanup failed"):
            delete(source.name)
        assert source.read_bytes() == b"recorded-result"
    delete(source.name)
    assert not source.exists()
    assert not artifact.exists()
    assert retained.read_bytes() == b"retained-result"


def test_summary_loader_skips_malformed_result_file(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.raw_dir.mkdir(parents=True)
    (service.raw_dir / "broken.csv").write_text("player,value\n0,1\n", encoding="utf-8")

    snapshot = service.result_snapshot()

    assert snapshot.summaries() == []
    assert len(snapshot.warnings) == 1
    assert "missing required columns" in snapshot.warnings[0]


def test_result_snapshot_reuses_unchanged_file_summary(tmp_path: Path, monkeypatch) -> None:
    from experiments import result_catalog
    service = create_service(tmp_path)
    run_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir, feedback_mode="full_information")
    summarize_file = result_catalog.load_final_result_rows
    calls = 0

    def counted_summary(path: Path):
        nonlocal calls
        calls += 1
        return summarize_file(path)

    monkeypatch.setattr(result_catalog, "load_final_result_rows", counted_summary)

    first = service.result_snapshot()
    second = service.result_snapshot()
    next(service.raw_dir.glob("*.csv")).unlink()
    after_delete = service.result_snapshot()

    assert calls == 1
    assert first.summaries() == second.summaries()
    assert [row["player"] for row in first.summaries()] == [0, 1]
    assert all(row["horizon"] == 2 for row in first.summaries())
    assert after_delete.filenames == ()
    assert after_delete.summaries() == []


@pytest.mark.parametrize("feedback_mode,algorithms,workers", [
    ("full_information", ("hedge", "hedge"), 1),
    ("bandit", ("auer_exp3", "lce_ix"), 2),
])
def test_submission_runs_requested_replicates(
    tmp_path: Path,
    feedback_mode: str,
    algorithms: tuple[str, str],
    workers: int,
) -> None:
    service = create_service(tmp_path)
    service.replicate_workers = workers
    form = ExperimentForm(
        "rps",
        feedback_mode,
        algorithms,
        horizon=2,
        seed=42,
        replicates=3,
    )

    job = service.submit_experiment(form)

    assert wait_for_job(service, job.id) == "succeeded"
    completed = service.jobs.get(job.id)
    assert (completed.completed, completed.total) == (3, 3)
    assert len(list(service.raw_dir.glob("*.csv"))) == 3
    assert not list(tmp_path.rglob("*.png")) and not list(tmp_path.rglob("*.pdf"))
    assert {
        summary["replicate"]
        for summary in service.result_snapshot().summaries()
    } == {0, 1, 2}


def test_experiment_submissions_queue_and_reserve_run_ids(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    blocker, release_blocker = block_job_queue(service.jobs)
    first = service.submit_experiment(experiment_form())

    with pytest.raises(FileExistsError, match="queued"):
        service.submit_experiment(experiment_form())

    second_form = ExperimentForm(
        "rps",
        "full_information",
        ("hedge", "hedge"),
        horizon=2,
        seed=43,
        replicates=1,
    )
    second = service.submit_experiment(second_form)

    assert service.jobs.get(first.id).status == "queued"
    assert service.jobs.get(second.id).status == "queued"
    release_blocker.set()
    assert wait_for_job(service, blocker.id) == "succeeded"
    assert wait_for_job(service, first.id) == "succeeded"
    assert wait_for_job(service, second.id) == "succeeded"
    assert len(list(service.raw_dir.glob("*.csv"))) == 2


def test_scaling_delete_rolls_back_csv_and_figures_when_rebuild_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = create_service(tmp_path)
    raw_dir = service.adversarial_scaling_raw_dir
    figure_dir = service.adversarial_scaling_figure_dir

    raw_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    csv_path = raw_dir / "result.csv"
    figure_path = figure_dir / "existing.png"
    csv_path.write_bytes(b"recorded-result")
    figure_path.write_bytes(b"existing-figure")
    monkeypatch.setattr(
        service,
        "_publish_adversarial_scaling_plots",
        lambda: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    with pytest.raises(PlotUpdateError, match="were restored"):
        service.delete_adversarial_scaling_experiment(csv_path.name)

    assert csv_path.read_bytes() == b"recorded-result"
    assert figure_path.read_bytes() == b"existing-figure"
    assert not list(raw_dir.parent.glob(".delete-result-*"))


@pytest.mark.parametrize("grouped", [False, True])
def test_joint_action_heatmap_is_generated_and_cached(tmp_path, grouped):
    service = create_service(tmp_path)
    paths = [run_cross_play_experiment("rps", ["hedge", "hedge"],
        horizon=3, replicate=r, output_dir=service.raw_dir, feedback_mode="full_information") for r in range(2 if grouped else 1)]
    if grouped:
        group_id = service.result_snapshot().summaries(grouped=True)[0]["group_id"]
        request = lambda: service.group_joint_action_figure(group_id)
    else:
        request = lambda: service.joint_action_figure(paths[0].name)
    first = request()
    timestamp = first.stat().st_mtime_ns
    assert request() == first and first.stat().st_mtime_ns == timestamp
    assert first.read_bytes().startswith(b"\x89PNG")
    assert first.with_suffix(".pdf").read_bytes().startswith(b"%PDF")


def test_custom_zero_sum_joint_action_heatmap_uses_saved_game(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("Joint Actions", 2, [3, 3], 7, "zero_sum")
    result_path = run_cross_play_experiment(
        definition.id,
        ["hedge", "hedge"],
        horizon=3,
        output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
        feedback_mode="full_information",
    )

    figure = service.joint_action_figure(result_path.name)

    assert figure.is_file()
    assert figure.with_suffix(".pdf").is_file()


@pytest.mark.parametrize("grouped", [False, True])
def test_equilibrium_distance_reuses_paired_cache_and_clears_it(tmp_path, monkeypatch, grouped):
    from experiments.plots import plot_equilibrium_convergence as plotting
    service = create_service(tmp_path)
    paths = [run_cross_play_experiment("rps", ["hedge", "hedge"],
        horizon=2, replicate=r, output_dir=service.raw_dir, feedback_mode="full_information") for r in range(2 if grouped else 1)]
    calls = []

    def render(input_paths, output_path, **kwargs):
        calls.append(list(input_paths))
        write_figure_pair(output_path, b"distance")

    monkeypatch.setattr(plotting, "plot_result_equilibrium_distance", render)
    if grouped:
        group_id = service.result_snapshot().summaries(grouped=True)[0]["group_id"]
        request = lambda: service.request_group_equilibrium_convergence_figure(group_id)
    else:
        request = lambda: service.request_equilibrium_convergence_figure(paths[0].name)
    first, error = wait_for_equilibrium_figure(request)
    assert error is None and first is not None
    assert wait_for_equilibrium_figure(request) == (first, None)
    assert len(calls) == 1  # Cache hits must not recompute distances.
    assert len(calls[0]) == len(paths) and set(calls[0]) == set(paths)
    assert first.read_bytes() == first.with_suffix(".pdf").read_bytes() == b"distance"
    service.clear_results()
    assert not first.exists() and not first.with_suffix(".pdf").exists()


@pytest.mark.parametrize("grouped", [False, True])
def test_annotated_equilibrium_figures_bypass_old_render_cache_without_deleting_it(tmp_path, monkeypatch, grouped):
    from experiments.plots import plot_equilibrium_convergence

    service = create_service(tmp_path)
    result_path = run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    if grouped:
        group_id = service.result_snapshot().summaries(grouped=True)[0]["group_id"]
        _, output_path, stem = service._group_convergence_figure_path(group_id)
        legacy_path = service.detail_figure_dir / f"{stem}_replicate_mean_equilibrium_distance.png"
        request_figure = lambda: service.request_group_equilibrium_convergence_figure(group_id)
    else:
        _, output_path = service._convergence_figure_path(result_path.name)
        legacy_path = service.detail_figure_dir / f"{result_path.stem}_equilibrium_distance.png"
        request_figure = lambda: service.request_equilibrium_convergence_figure(result_path.name)
    service.detail_figure_dir.mkdir(parents=True, exist_ok=True)
    write_figure_pair(legacy_path, b"old unannotated figure")
    calls = []

    def fake_plot(input_paths, path, **kwargs):
        calls.append(input_paths)
        write_figure_pair(path, b"annotated figure")

    monkeypatch.setattr(plot_equilibrium_convergence, "plot_result_equilibrium_distance", fake_plot)
    generated, error = wait_for_equilibrium_figure(request_figure)
    assert error is None and generated == output_path
    assert generated != legacy_path
    assert generated.read_bytes() == b"annotated figure"
    assert generated.with_suffix(".pdf").read_bytes() == b"annotated figure"
    assert legacy_path.read_bytes() == legacy_path.with_suffix(".pdf").read_bytes() == b"old unannotated figure"
    assert wait_for_equilibrium_figure(request_figure) == (generated, None)
    assert len(calls) == 1


def test_dashboard_keeps_active_jobs_outside_display_limit(tmp_path, monkeypatch):
    app, service = create_test_app(tmp_path)
    jobs = [Job(str(i), "Finished", "succeeded", "Done", "now") for i in range(5)]
    jobs += [Job("active", "Active", "running", "Running", "now"), Job("old", "Old", "failed", "Failed", "now")]
    monkeypatch.setattr(service.jobs, "recent", lambda: jobs)
    response = app.test_client().get("/")
    page = response.get_data(as_text=True)
    for job in jobs:
        assert (f'data-job-id="{job.id}"' in page) == (job.id != "old")
