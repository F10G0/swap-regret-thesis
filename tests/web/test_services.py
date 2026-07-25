from pathlib import Path
import logging
from threading import Event
import time

import pytest

from experiments.scenarios.bandit_cross_play import (
    run_bandit_cross_play_experiment,
)
from experiments.scenarios.full_information_cross_play import (
    run_full_information_cross_play_experiment,
)
from web.services import (
    DashboardService,
    GAME_PRESENTATIONS,
    JobManager,
    ServiceBusyError,
)
from web.validation import ExperimentForm


def create_service(tmp_path: Path) -> DashboardService:
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
    )
    service._publish_plots = lambda game_name=None: None
    return service


def wait_for_job(service: DashboardService, job_id: str) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.jobs.get(job_id)
        if job is not None and job.status in {"succeeded", "failed", "cancelled"}:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def experiment_form() -> ExperimentForm:
    return ExperimentForm(
        game="rps",
        feedback_mode="full_information",
        algorithm_player_0="hedge",
        algorithm_player_1="hedge",
        horizon=2,
        seed=42,
        replicate=0,
        replicates=1,
    )


def test_bandit_dashboard_exposes_lce_ix() -> None:
    service = DashboardService(results_dir="results", raw_dir="results/raw", figure_dir="results/figures")

    assert "lce_ix" in service.algorithms_by_feedback_mode["bandit"]
    assert service.algorithm_labels["lce_ix"] == "LCE-IX"


def test_dashboard_exposes_presentations_for_all_bertrand_benchmarks() -> None:
    service = DashboardService(
        results_dir="results",
        raw_dir="results/raw",
        figure_dir="results/figures",
    )
    bertrand_games = {
        "bertrand_standard_o1",
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    }

    assert bertrand_games <= set(service.games)
    assert set(service.game_presentations) == set(service.games)
    for game_name in bertrand_games:
        assert service.game_presentations[game_name] == GAME_PRESENTATIONS[game_name]


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

    terminal_statuses = {}
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        terminal_statuses = {
            job_id: manager.get(job_id).status
            for job_id in (first.id, second.id)
        }
        if set(terminal_statuses.values()) == {"succeeded"}:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"queued jobs did not complete: {terminal_statuses}")

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
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if manager.get(first.id).status == "succeeded" and manager.get(replacement.id).status == "succeeded":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("replacement job did not complete")


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
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if manager.get(job.id).status == "succeeded":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("active job did not complete")


def test_job_manager_records_and_logs_failures(caplog) -> None:
    manager = JobManager()
    caplog.set_level(logging.ERROR, logger="web.services")

    def operation(job) -> None:
        raise RuntimeError("failed operation")

    job = manager.submit("failure", operation)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        failed_job = manager.get(job.id)
        if failed_job is not None and failed_job.status == "failed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("job did not fail")

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

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = manager.get(submitted.id)
        if job.status == "cancelled":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("job was not cancelled")

    assert job.completed == 1
    assert job.total == 4


def test_plot_rebuild_job_does_not_request_page_reload(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    job = service.submit_plot_rebuild()

    assert wait_for_job(service, job.id) == "succeeded"
    assert service.jobs.get(job.id).reload_page is False


def test_clear_results_preserves_unrelated_files(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.raw_dir.mkdir(parents=True)
    service.figure_dir.mkdir(parents=True)
    (service.raw_dir / ".gitkeep").write_text("", encoding="utf-8")
    (service.raw_dir / "run.csv").write_text("generated", encoding="utf-8")
    (service.raw_dir / "notes.txt").write_text("keep", encoding="utf-8")
    (service.figure_dir / "figure.png").write_bytes(b"generated")
    (service.results_dir / "index.html").write_text("generated", encoding="utf-8")

    service.clear_results()

    assert (service.raw_dir / ".gitkeep").exists()
    assert (service.raw_dir / "notes.txt").exists()
    assert not (service.raw_dir / "run.csv").exists()
    assert not (service.figure_dir / "figure.png").exists()
    assert not (service.results_dir / "index.html").exists()


def test_summary_loader_skips_malformed_result_file(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.raw_dir.mkdir(parents=True)
    (service.raw_dir / "broken.csv").write_text("player,value\n0,1\n", encoding="utf-8")

    summaries, warnings = service.experiment_summaries()

    assert summaries == []
    assert len(warnings) == 1
    assert "missing required columns" in warnings[0]


def test_summary_loader_returns_final_rows_for_each_player(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=2,
        output_dir=service.raw_dir,
    )

    summaries, warnings = service.experiment_summaries()

    assert warnings == []
    assert [summary["player"] for summary in summaries] == [0, 1]
    assert all(summary["horizon"] == 2 for summary in summaries)


def test_result_snapshot_reuses_unchanged_file_summary(tmp_path: Path, monkeypatch) -> None:
    service = create_service(tmp_path)
    run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)
    summarize_file = service.result_index._summarize_file
    calls = 0

    def counted_summary(path: Path):
        nonlocal calls
        calls += 1
        return summarize_file(path)

    monkeypatch.setattr(service.result_index, "_summarize_file", counted_summary)

    first = service.result_snapshot()
    second = service.result_snapshot()
    next(service.raw_dir.glob("*.csv")).unlink()
    after_delete = service.result_snapshot()

    assert calls == 1
    assert first == second
    assert after_delete.filenames == []
    assert after_delete.summaries == []


def test_all_pairs_skips_existing_runs_on_retry(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    form = experiment_form()
    pair_count = len(service.algorithms_by_feedback_mode["full_information"]) ** 2

    first_job, scheduled, skipped = service.submit_all_pairs(form)
    assert (scheduled, skipped) == (pair_count, 0)
    assert wait_for_job(service, first_job.id) == "succeeded"

    first_output = sorted(service.raw_dir.glob("*.csv"))[0]
    first_output.unlink()

    retry_job, scheduled, skipped = service.submit_all_pairs(form)
    assert (scheduled, skipped) == (1, pair_count - 1)
    assert wait_for_job(service, retry_job.id) == "succeeded"
    assert len(list(service.raw_dir.glob("*.csv"))) == pair_count


def test_bandit_submission_runs_requested_replicates(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    form = ExperimentForm("rps", "bandit", "exp3", "lce_ix", horizon=2, seed=42, replicate=3, replicates=3)

    job = service.submit_experiment(form)

    assert wait_for_job(service, job.id) == "succeeded"
    completed = service.jobs.get(job.id)
    assert (completed.completed, completed.total) == (3, 3)
    assert len(list(service.raw_dir.glob("*.csv"))) == 3


def test_experiment_submissions_queue_and_reserve_run_ids(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    blocker_started = Event()
    release_blocker = Event()

    def block_queue(job) -> None:
        blocker_started.set()
        assert release_blocker.wait(timeout=2)

    blocker = service.jobs.submit("blocker", block_queue)
    assert blocker_started.wait(timeout=1)
    first = service.submit_experiment(experiment_form())

    with pytest.raises(FileExistsError, match="queued"):
        service.submit_experiment(experiment_form())

    second_form = ExperimentForm("rps", "full_information", "hedge", "hedge", horizon=2, seed=43, replicate=0, replicates=1)
    second = service.submit_experiment(second_form)

    assert service.jobs.get(first.id).status == "queued"
    assert service.jobs.get(second.id).status == "queued"
    release_blocker.set()
    assert wait_for_job(service, blocker.id) == "succeeded"
    assert wait_for_job(service, first.id) == "succeeded"
    assert wait_for_job(service, second.id) == "succeeded"
    assert len(list(service.raw_dir.glob("*.csv"))) == 2


def test_plot_publication_creates_structured_figure_metadata(tmp_path: Path) -> None:
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
    )
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=2,
        output_dir=service.raw_dir,
    )

    service._publish_plots("rps")
    figures = service.figure_records()

    assert len(figures) == 12
    assert {figure["source"] for figure in figures} == {"expected"}
    assert {figure["regret"] for figure in figures} == {"external", "internal", "swap"}
    assert {figure["player"] for figure in figures} == {0, 1}
    assert {figure["view"] for figure in figures} == {"average", "sqrt_scaling"}


def test_failed_plot_update_preserves_existing_figures(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.figure_dir.mkdir(parents=True)
    existing_figure = service.figure_dir / "rps_average_expected_external_regret_player_0.png"
    existing_figure.write_bytes(b"existing")

    def fail_plot_update(game_name=None) -> None:
        raise RuntimeError("plot failed")

    service._publish_plots = fail_plot_update

    job = service.submit_experiment(experiment_form())

    assert wait_for_job(service, job.id) == "failed"
    assert existing_figure.read_bytes() == b"existing"


def test_plot_publication_skips_malformed_result_files(tmp_path: Path) -> None:
    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)
    (service.raw_dir / "broken.csv").write_text("player,value\n0,1\n", encoding="utf-8")

    service._publish_plots("rps")

    assert len(service.figure_records()) == 12


def test_selected_game_plotting_ignores_unrelated_malformed_results(tmp_path: Path) -> None:
    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)
    (service.raw_dir / "unrelated_broken.csv").write_text("invalid\nvalue\n", encoding="utf-8")

    service._publish_plots("rps")

    assert len(service.figure_records()) == 12


def test_bandit_results_contain_only_realized_regret(tmp_path: Path) -> None:
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
    )
    run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["exp3", "exp3"],
        horizon=2,
        output_dir=service.raw_dir,
    )

    summaries, warnings = service.experiment_summaries()
    service._publish_plots("rps")
    figures = service.figure_records()

    assert warnings == []
    assert len(summaries) == 2
    assert all("average_realized_swap_regret" in summary for summary in summaries)
    assert all("average_expected_swap_regret" not in summary for summary in summaries)
    assert len(figures) == 12
    assert {figure["source"] for figure in figures} == {"realized"}


def test_plotting_keeps_expected_and_realized_results_for_the_same_game(tmp_path: Path) -> None:
    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)
    run_bandit_cross_play_experiment(game_name="rps", algorithm_names=["exp3", "exp3"], horizon=2, output_dir=service.raw_dir)

    service._publish_plots("rps")
    figures = service.figure_records()

    assert len(figures) == 24
    assert {figure["source"] for figure in figures} == {"expected", "realized"}


def test_joint_action_heatmap_is_generated_and_cached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from matplotlib.axes import Axes

    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    result_path = run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=3, output_dir=service.raw_dir)
    original_imshow = Axes.imshow
    colormaps = []

    def capture_colormap(axes, *args, **kwargs):
        colormaps.append(kwargs.get("cmap"))
        return original_imshow(axes, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_colormap)

    first = service.joint_action_figure(result_path.name)
    first_timestamp = first.stat().st_mtime_ns
    second = service.joint_action_figure(result_path.name)

    assert first == second
    assert second.stat().st_mtime_ns == first_timestamp
    assert second.stat().st_size > 0
    assert colormaps == ["Blues"]


def test_equilibrium_heatmap_uses_png_as_its_only_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from matplotlib.axes import Axes

    from experiments.plots import plot_equilibrium_weights

    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
    )
    original_plot = (
        plot_equilibrium_weights.plot_equilibrium_profile_weights
    )
    calls = 0
    original_imshow = Axes.imshow
    colormaps = []

    def capture_colormap(axes, *args, **kwargs):
        colormaps.append(kwargs.get("cmap"))
        return original_imshow(axes, *args, **kwargs)

    def counted_plot(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        original_plot(*args, **kwargs)

    monkeypatch.setattr(
        plot_equilibrium_weights,
        "plot_equilibrium_profile_weights",
        counted_plot,
    )
    monkeypatch.setattr(Axes, "imshow", capture_colormap)

    first = service.equilibrium_figure("rps", "ce")
    first_timestamp = first.stat().st_mtime_ns
    second = service.equilibrium_figure("rps", "ce")

    assert calls == 1
    assert first == second
    assert "_blue_" in first.name
    assert second.stat().st_mtime_ns == first_timestamp
    assert second.stat().st_size > 0
    assert colormaps == ["Blues"]
    assert service.figure_records() == []

    service.clear_results()
    assert not first.exists()


@pytest.mark.parametrize(
    ("game_name", "equilibrium"),
    [
        ("unknown", "ce"),
        ("rps", "nash"),
    ],
)
def test_equilibrium_heatmap_rejects_unknown_parameters(
    tmp_path: Path,
    game_name: str,
    equilibrium: str,
) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError):
        service.equilibrium_figure(game_name, equilibrium)
