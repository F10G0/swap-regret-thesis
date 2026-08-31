from pathlib import Path
import logging
from threading import Event
import time

import numpy as np
import pytest

from experiments.scenarios.bandit_cross_play import (
    run_bandit_cross_play_experiment,
)
from experiments.scenarios.full_information_cross_play import (
    run_full_information_cross_play_experiment,
)
from web.jobs import JobManager, ServiceBusyError
from web.presentations import GAME_PRESENTATIONS
from web.result_groups import aggregate_result_summaries
from web.services import DashboardService
from tests.web.support import block_job_queue, create_service, wait_for_async_result, wait_for_job
from experimental.equilibrium_trajectory.web_models import (
    comparison_member_colors,
    stable_member_color,
)
from web.validation import ExperimentForm


def write_figure_pair(output_path, content: bytes) -> None:
    output_path = Path(output_path)
    output_path.write_bytes(content)
    output_path.with_suffix(".pdf").write_bytes(content)


def wait_for_trajectory_comparison(
    service: DashboardService,
    group_ids: list[str],
    final_interval_segments: int = 4,
    focus_final_interval: bool = False,
    comparison_view: str = "geometry",
):
    return wait_for_async_result(
        lambda: service.experimental_trajectory.request(
            group_ids,
            final_interval_segments,
            focus_final_interval,
            comparison_view,
        )
    )


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


def test_bandit_dashboard_exposes_lce_ix() -> None:
    service = DashboardService(results_dir="results", raw_dir="results/raw", figure_dir="results/figures")

    assert "lce_ix" in service.algorithms_by_feedback_mode["bandit"]
    assert service.algorithm_labels["lce_ix"] == "LCE-IX"
    assert service.algorithm_labels["regret_matching"] == "RM"
    assert service.algorithm_labels["stationary_regret_matching"] == "SRM"


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


def test_plot_rebuild_job_completes(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    job = service.submit_plot_rebuild()

    assert wait_for_job(service, job.id) == "succeeded"


def test_clear_results_preserves_unrelated_files(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.raw_dir.mkdir(parents=True)
    service.figure_dir.mkdir(parents=True)
    (service.raw_dir / ".gitkeep").write_text("", encoding="utf-8")
    (service.raw_dir / "run.csv").write_text("generated", encoding="utf-8")
    (service.raw_dir / "notes.txt").write_text("keep", encoding="utf-8")
    (service.figure_dir / "figure.png").write_bytes(b"generated")

    service.clear_results()

    assert (service.raw_dir / ".gitkeep").exists()
    assert (service.raw_dir / "notes.txt").exists()
    assert not (service.raw_dir / "run.csv").exists()
    assert not (service.figure_dir / "figure.png").exists()


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


def test_custom_game_inspection_and_payoff_slice(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("inspect me", 3, [2, 3, 2], 7)
    payoff_tensor = service.game_catalog.load(definition.id)

    inspection = service.custom_game_inspection(definition.id)
    payoff_slice = service.custom_game_payoff_slice(definition.id, 2, 1, 0, [0, 0, 1])

    assert inspection["shape"] == (3, 2, 3, 2)
    assert inspection["minimum"] == pytest.approx(np.min(payoff_tensor))
    assert inspection["maximum"] == pytest.approx(np.max(payoff_tensor))
    assert inspection["mean"] == pytest.approx(np.mean(payoff_tensor))
    assert np.array_equal(payoff_slice["values"], payoff_tensor[2, :, :, 1].T)
    assert service.custom_game_file(definition.id).name == "inspect-me.npz"


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


def test_summary_loader_skips_malformed_result_file(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.raw_dir.mkdir(parents=True)
    (service.raw_dir / "broken.csv").write_text("player,value\n0,1\n", encoding="utf-8")

    snapshot = service.result_snapshot()

    assert snapshot.summaries == []
    assert len(snapshot.warnings) == 1
    assert "missing required columns" in snapshot.warnings[0]


def test_summary_loader_returns_final_rows_for_each_player(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=2,
        output_dir=service.raw_dir,
    )
    snapshot = service.result_snapshot()

    assert snapshot.warnings == []
    assert [summary["player"] for summary in snapshot.summaries] == [0, 1]
    assert all(summary["horizon"] == 2 for summary in snapshot.summaries)


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


@pytest.mark.parametrize(
    ("feedback_mode", "algorithms"),
    [
        ("full_information", ("hedge", "hedge")),
        ("bandit", ("exp3", "lce_ix")),
    ],
)
def test_submission_runs_requested_replicates(
    tmp_path: Path,
    feedback_mode: str,
    algorithms: tuple[str, str],
) -> None:
    service = create_service(tmp_path)
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
    assert {
        summary["replicate"]
        for summary in service.result_snapshot().summaries
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
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=2,
        replicate=1,
        output_dir=service.raw_dir,
    )

    service._publish_plots("rps")
    figures = service.figure_records()

    assert len(figures) == 12
    assert {figure["source"] for figure in figures} == {"expected"}
    assert {figure["regret"] for figure in figures} == {"external", "internal", "swap"}
    assert {figure["player"] for figure in figures} == {0, 1}
    assert {figure["view"] for figure in figures} == {"average", "sqrt_scaling"}
    assert all((service.figure_dir / figure["pdf_filename"]).is_file() for figure in figures)
    assert all(figure["confidence_free_filename"] != figure["filename"] for figure in figures)
    assert all((service.figure_dir / figure["confidence_free_filename"]).is_file() for figure in figures)
    assert all((service.figure_dir / figure["confidence_free_pdf_filename"]).is_file() for figure in figures)
    assert service.validate_figure_filename(figures[0]["confidence_free_filename"])


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

    snapshot = service.result_snapshot()
    service._publish_plots("rps")
    figures = service.figure_records()

    assert snapshot.warnings == []
    assert len(snapshot.summaries) == 2
    assert all("average_realized_swap_regret" in summary for summary in snapshot.summaries)
    assert all("average_expected_swap_regret" not in summary for summary in snapshot.summaries)
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


def test_both_regret_evaluations_are_summarized_and_plotted(tmp_path: Path) -> None:
    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    run_full_information_cross_play_experiment(
        game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2,
        output_dir=service.raw_dir, regret_evaluation="both",
    )

    snapshot = service.result_snapshot()
    service._publish_plots("rps")
    figures = service.figure_records()

    assert snapshot.warnings == []
    assert {summary["regret_evaluation"] for summary in snapshot.summaries} == {"both"}
    assert all("average_expected_swap_regret" in summary for summary in snapshot.summaries)
    assert all("average_realized_swap_regret" in summary for summary in snapshot.summaries)
    assert len(figures) == 24
    assert {figure["source"] for figure in figures} == {"expected", "realized"}
    assert sum(figure["source"] == "expected" for figure in figures) == 12
    assert sum(figure["source"] == "realized" for figure in figures) == 12
    assert all("both" not in figure["filename"] for figure in figures)


def test_joint_action_heatmap_is_generated_and_cached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from matplotlib.axes import Axes

    service = DashboardService(results_dir=tmp_path, raw_dir=tmp_path / "raw", figure_dir=tmp_path / "figures")
    result_path = run_full_information_cross_play_experiment(game_name="rps", algorithm_names=["hedge", "hedge"], horizon=3, output_dir=service.raw_dir)
    original_imshow = Axes.imshow
    colormaps = []
    origins = []

    def capture_colormap(axes, *args, **kwargs):
        colormaps.append(kwargs.get("cmap"))
        origins.append(kwargs.get("origin"))
        return original_imshow(axes, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_colormap)

    first = service.joint_action_figure(result_path.name)
    first_timestamp = first.stat().st_mtime_ns
    second = service.joint_action_figure(result_path.name)

    assert first == second
    assert first.name.endswith("_joint_actions_blue_lower_origin.png")
    assert second.stat().st_mtime_ns == first_timestamp
    assert second.stat().st_size > 0
    assert second.with_suffix(".pdf").is_file()
    assert colormaps == ["Blues"]
    assert origins == ["lower"]


def test_replicate_group_joint_action_heatmap_is_generated_and_cached(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    for replicate in range(2):
        run_full_information_cross_play_experiment(
            game_name="rps", algorithm_names=["hedge", "hedge"], horizon=3, replicate=replicate, output_dir=service.raw_dir
        )
    group_id = aggregate_result_summaries(service.result_snapshot().summaries)[0]["group_id"]

    first = service.group_joint_action_figure(group_id)
    first_timestamp = first.stat().st_mtime_ns
    second = service.group_joint_action_figure(group_id)

    assert first == second
    assert first.name.endswith("_replicate_mean_joint_actions_blue_lower_origin.png")
    assert second.stat().st_mtime_ns == first_timestamp
    assert second.stat().st_size > 0
    assert second.with_suffix(".pdf").is_file()


def test_custom_zero_sum_joint_action_heatmap_uses_saved_game(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("Joint Actions", 2, [3, 3], 7, "zero_sum")
    result_path = run_full_information_cross_play_experiment(
        definition.id,
        ["hedge", "hedge"],
        horizon=3,
        output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
    )

    figure = service.joint_action_figure(result_path.name)

    assert figure.is_file()
    assert figure.with_suffix(".pdf").is_file()


def test_equilibrium_convergence_figures_share_computation_and_use_paired_cache(tmp_path: Path, monkeypatch) -> None:
    from experiments.plots import plot_equilibrium_convergence

    service = create_service(tmp_path)
    result_path = run_full_information_cross_play_experiment(
        game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir
    )
    calls = 0

    def fake_plot(input_path, distance_output_path, **kwargs) -> None:
        nonlocal calls
        calls += 1
        write_figure_pair(distance_output_path, b"distance")

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "plot_result_equilibrium_distance",
        fake_plot,
    )

    request_figure = lambda: service.request_equilibrium_convergence_figure(result_path.name)
    first_path, first_error = wait_for_equilibrium_figure(request_figure)
    second_path, second_error = wait_for_equilibrium_figure(request_figure)

    assert calls == 1
    assert first_error is second_error is None
    assert first_path == second_path
    assert first_path.read_bytes() == b"distance"
    assert first_path.with_suffix(".pdf").read_bytes() == b"distance"

    service.clear_results()
    assert not first_path.exists()


def test_replicate_group_equilibrium_figures_share_computation_and_use_paired_cache(tmp_path: Path, monkeypatch) -> None:
    from experiments.plots import plot_equilibrium_convergence

    service = create_service(tmp_path)
    for replicate in range(2):
        run_full_information_cross_play_experiment(
            game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, replicate=replicate, output_dir=service.raw_dir
        )
    group_id = aggregate_result_summaries(service.result_snapshot().summaries)[0]["group_id"]
    received_paths = []

    def fake_plot(input_paths, distance_output_path, **kwargs) -> None:
        received_paths.extend(input_paths)
        write_figure_pair(distance_output_path, b"mean distance")

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "plot_result_equilibrium_distance",
        fake_plot,
    )

    request_figure = lambda: service.request_group_equilibrium_convergence_figure(group_id)
    first_path, first_error = wait_for_equilibrium_figure(request_figure)
    second_path, second_error = wait_for_equilibrium_figure(request_figure)

    assert len(received_paths) == 2
    assert {path.name for path in received_paths} == set(service.result_snapshot().filenames)
    assert first_error is second_error is None
    assert first_path == second_path
    assert first_path.read_bytes() == b"mean distance"
    assert first_path.with_suffix(".pdf").read_bytes() == b"mean distance"


def test_core_distance_generation_does_not_load_experimental_trajectory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experiments.plots import plot_equilibrium_convergence

    service = create_service(tmp_path)
    result_path = run_full_information_cross_play_experiment(
        game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir
    )
    calls = 0

    def fake_distance(input_paths, output_path, **kwargs) -> None:
        nonlocal calls
        calls += 1
        write_figure_pair(output_path, b"trajectory 20")

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "plot_result_equilibrium_distance",
        fake_distance,
    )

    assert service._experimental_trajectory_dashboard is None
    generated, error = wait_for_equilibrium_figure(
        lambda: service.request_equilibrium_convergence_figure(result_path.name)
    )

    assert calls == 1
    assert error is None
    assert generated is not None
    assert service._experimental_trajectory_dashboard is None


def test_trajectory_comparison_candidates_expose_exact_replicate_seed_protocol(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)
    for algorithms in (
        ["hedge", "hedge"],
        ["regret_matching", "regret_matching"],
    ):
        for replicate in (0, 1):
            run_full_information_cross_play_experiment(
                game_name="rps",
                algorithm_names=algorithms,
                horizon=3,
                seed=42,
                replicate=replicate,
                output_dir=service.raw_dir,
            )

    candidates = service.experimental_trajectory.candidates()

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate["replicate_count"] == 2
        assert candidate["replicate_indices"] == [0, 1]
        assert candidate["player_seed_schedule"] == [
            [42, 43],
            [44, 45],
        ]
        assert candidate["compatibility_key"][-3:] == (
            2,
            (0, 1),
            ((42, 43), (44, 45)),
        )


def test_comparison_colors_are_order_independent_and_high_contrast() -> None:
    group_ids = ["f", "a", "d", "b", "e", "c"]

    colors = comparison_member_colors(group_ids)
    reversed_colors = comparison_member_colors(reversed(group_ids))

    assert colors == reversed_colors
    assert list(colors) == sorted(group_ids)
    assert list(colors.values()) == [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    assert len(set(colors.values())) == len(group_ids)
    assert comparison_member_colors(["only"]) == {
        "only": stable_member_color("only")
    }
    overflow_colors = comparison_member_colors(
        f"member-{position:02d}" for position in range(24)
    )
    assert len(set(overflow_colors.values())) == 24


def test_trajectory_comparison_rejects_different_replicate_populations(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)
    for replicate in (0, 1):
        run_full_information_cross_play_experiment(
            game_name="rps",
            algorithm_names=["hedge", "hedge"],
            horizon=3,
            seed=42,
            replicate=replicate,
            output_dir=service.raw_dir,
        )
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["regret_matching", "regret_matching"],
        horizon=3,
        seed=42,
        replicate=0,
        output_dir=service.raw_dir,
    )
    group_ids = [
        candidate["group_id"]
        for candidate in service.experimental_trajectory.candidates()
    ]

    with pytest.raises(ValueError, match="replicate count"):
        service.experimental_trajectory.request(group_ids)


def test_trajectory_comparison_cache_is_order_independent_and_colors_are_authoritative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import dashboard as trajectory_dashboard

    service = create_service(tmp_path)
    for algorithms in (
        ["hedge", "hedge"],
        ["regret_matching", "regret_matching"],
    ):
        run_full_information_cross_play_experiment(
            game_name="rps",
            algorithm_names=algorithms,
            horizon=3,
            seed=42,
            replicate=0,
            output_dir=service.raw_dir,
        )
    candidates = service.experimental_trajectory.candidates()
    group_ids = [candidate["group_id"] for candidate in candidates]
    captured_members = []

    def fake_comparison(members, output_path, **kwargs):
        captured_members.extend(members)
        write_figure_pair(output_path, b"comparison")

    monkeypatch.setattr(
        trajectory_dashboard,
        "plot_result_equilibrium_trajectory_comparison",
        fake_comparison,
    )
    forward = service.experimental_trajectory.definition(
        group_ids,
        4,
        True,
    )
    reverse = service.experimental_trajectory.definition(
        list(reversed(group_ids)),
        4,
        True,
    )
    result, error = wait_for_trajectory_comparison(
        service,
        list(reversed(group_ids)),
        final_interval_segments=4,
        focus_final_interval=True,
    )

    assert error is None
    assert result is not None
    assert forward.artifact_id == reverse.artifact_id
    assert [member.group_id for member in forward.members] == sorted(group_ids)
    assert [member.member_id for member in captured_members] == sorted(group_ids)
    assert [member.color for member in captured_members] == [
        "#1f77b4",
        "#ff7f0e",
    ]
    response_members = result.public_data("/comparison.png")["members"]
    assert [member["group_id"] for member in response_members] == sorted(group_ids)
    assert [member["color"] for member in response_members] == [
        member.color for member in captured_members
    ]
    assert result.output_path.with_suffix(".pdf").is_file()
    assert result.output_path.read_bytes() == b"comparison"


def test_trajectory_comparison_views_have_distinct_cache_ids_and_share_colors(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)
    for algorithms in (
        ["hedge", "hedge"],
        ["regret_matching", "regret_matching"],
    ):
        run_full_information_cross_play_experiment(
            game_name="rps",
            algorithm_names=algorithms,
            horizon=3,
            seed=42,
            replicate=0,
            output_dir=service.raw_dir,
        )
    group_ids = [
        candidate["group_id"]
        for candidate in service.experimental_trajectory.candidates()
    ]

    default_geometry = service.experimental_trajectory.definition(
        group_ids,
        4,
        True,
    )
    explicit_geometry = service.experimental_trajectory.definition(
        list(reversed(group_ids)),
        4,
        True,
        "geometry",
    )
    unified = service.experimental_trajectory.definition(
        group_ids,
        4,
        True,
        "unified",
    )

    assert default_geometry.artifact_id == explicit_geometry.artifact_id
    assert unified.artifact_id != explicit_geometry.artifact_id
    assert default_geometry.comparison_view == "geometry"
    assert unified.comparison_view == "unified"
    assert [member.color for member in unified.members] == [
        member.color for member in default_geometry.members
    ]


def test_unified_service_generation_passes_view_without_using_geometry_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import dashboard as trajectory_dashboard

    service = create_service(tmp_path)
    run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=3,
        seed=42,
        replicate=0,
        output_dir=service.raw_dir,
    )
    group_id = service.experimental_trajectory.candidates()[0]["group_id"]
    received = []

    def fake_comparison(members, output_path, **kwargs):
        received.append(kwargs)
        write_figure_pair(output_path, b"unified")

    monkeypatch.setattr(
        trajectory_dashboard,
        "plot_result_equilibrium_trajectory_comparison",
        fake_comparison,
    )

    result, error = wait_for_trajectory_comparison(
        service,
        [group_id],
        comparison_view="unified",
    )

    assert error is None
    assert result is not None
    assert result.definition.comparison_view == "unified"
    assert received[0]["comparison_view"] == "unified"
    assert result.public_data("/comparison.png")["comparison_view"] == "unified"


def test_equilibrium_distance_is_available_before_projected_regions_finish(tmp_path: Path, monkeypatch) -> None:
    from experiments.plots import plot_equilibrium_convergence
    from experimental.equilibrium_trajectory import dashboard as trajectory_dashboard

    service = create_service(tmp_path)
    result_path = run_full_information_cross_play_experiment(
        game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, output_dir=service.raw_dir
    )
    trajectory_started = Event()
    release_trajectory = Event()

    def plot_distance(input_path, output_path, **kwargs) -> None:
        write_figure_pair(output_path, b"distance")

    def plot_trajectory(members, output_path, **kwargs) -> None:
        trajectory_started.set()
        assert release_trajectory.wait(timeout=2)
        write_figure_pair(output_path, b"trajectory")

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "plot_result_equilibrium_distance",
        plot_distance,
    )
    monkeypatch.setattr(
        trajectory_dashboard,
        "plot_result_equilibrium_trajectory_comparison",
        plot_trajectory,
    )

    group_id = service.experimental_trajectory.candidates()[0]["group_id"]
    assert service.experimental_trajectory.request([group_id]) == (
        None,
        None,
    )
    assert trajectory_started.wait(timeout=1)

    first_path, first_error = service.request_equilibrium_convergence_figure(result_path.name)
    assert (first_path, first_error) == (None, None)

    distance_path, distance_error = wait_for_async_result(
        lambda: service.request_equilibrium_convergence_figure(result_path.name),
        timeout=1,
    )
    assert distance_error is None
    assert distance_path is not None
    assert distance_path.read_bytes() == b"distance"

    release_trajectory.set()
    result, trajectory_error = wait_for_async_result(
        lambda: service.experimental_trajectory.request([group_id]),
        timeout=2,
    )
    assert trajectory_error is None
    assert result.output_path.read_bytes() == b"trajectory"


def test_equilibrium_heatmap_uses_persistent_precomputed_asset(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    first = service.equilibrium_figure("rps", "ce")
    second = service.equilibrium_figure("rps", "ce")

    assert first == second
    assert first.name.endswith("_blue_lower_origin_maximum_profile_weight.png")
    assert first.stat().st_size > 0
    assert service.figure_records() == []

    service.clear_results()
    assert first.exists()


def test_custom_zero_sum_equilibrium_heatmap_is_cached_with_game(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experiments.plots import plot_equilibrium_weights

    service = create_service(tmp_path)
    definition = service.create_custom_game(
        "Cached Zero Sum",
        2,
        [2, 2],
        9,
        "zero_sum",
    )
    calls = []

    def fake_plot(payoff_tensor, equilibrium, output_path, game_name=None):
        calls.append((payoff_tensor.copy(), equilibrium, game_name))
        write_figure_pair(output_path, b"cached heatmap")

    monkeypatch.setattr(
        plot_equilibrium_weights,
        "plot_equilibrium_profile_weights",
        fake_plot,
    )

    first = service.equilibrium_figure(definition.id, "ce")
    second = service.equilibrium_figure(definition.id, "ce")
    pdf = service.equilibrium_figure(definition.id, "ce", "pdf")
    restarted_service = create_service(tmp_path)
    third = restarted_service.equilibrium_figure(definition.id, "ce")

    assert service.supports_matrix_figures(definition.id)
    assert first == second == third
    assert pdf == first.with_suffix(".pdf")
    assert pdf.read_bytes() == b"cached heatmap"
    assert first.parent == tmp_path / "custom-games" / ".equilibria"
    assert first.read_bytes() == b"cached heatmap"
    assert len(calls) == 1
    assert calls[0][1:] == ("ce", "Cached Zero Sum")

    service.clear_results()
    assert first.exists()

    service.delete_custom_game(definition.id)
    assert not first.exists()


def test_custom_general_sum_game_has_no_equilibrium_heatmap(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("General Sum", 2, [2, 2], 4)

    assert not service.supports_matrix_figures(definition.id)
    with pytest.raises(ValueError):
        service.equilibrium_figure(definition.id, "ce")


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
