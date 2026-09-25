from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import shutil
from threading import Event, current_thread
import time

import pytest
from pypdf import PdfReader

from web.jobs import Job, JobManager, ServiceBusyError
from experiments.scenarios.adversarial import run_adversarial_experiment
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.web.support import block_job_queue, browse_url, create_service, create_test_app, csrf_token, wait_for_async_result, wait_for_job
from web.validation import AdversarialExperimentForm, ExperimentForm, parse_profile_selection


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
        job.update_round_progress(25, 120)
        started.set()
        while not job.cancelled:
            time.sleep(0.001)
        job.check_cancelled()

    submitted = manager.submit("cancellable", operation, total=4, rounds_total=100)
    assert started.wait(timeout=1)
    manager.cancel(submitted.id)

    assert wait_for_job(manager, submitted.id) == "cancelled"
    job = manager.get(submitted.id)

    assert job.completed == 1
    assert job.total == 4
    assert (job.rounds_completed, job.rounds_total, job.eta_seconds) == (25, 100, 120)


def test_detailed_progress_threshold_counts_only_missing_runs(tmp_path, monkeypatch) -> None:
    service = create_service(tmp_path)
    callbacks = []

    def run_replicates_without_work(function, tasks, *, completed, round_progress, **kwargs):
        callbacks.append(round_progress is not None)
        if round_progress is not None:
            round_progress(sum(task["horizon"] for task in tasks))
        for _ in tasks:
            completed()
        return []

    monkeypatch.setattr("web.services.run_replicates", run_replicates_without_work)
    below = service.submit_experiment(ExperimentForm("rps", "full_information", ("hedge", "hedge"), 99_999, 1, 1))
    assert below.rounds_total == 0
    assert wait_for_job(service, below.id) == "succeeded"

    threshold = service.submit_experiment(ExperimentForm("rps", "full_information", ("hedge", "hedge"), 50_000, 2, 2))
    assert threshold.rounds_total == 100_000
    assert wait_for_job(service, threshold.id) == "succeeded"
    completed = service.jobs.get(threshold.id)
    assert (completed.rounds_completed, completed.eta_seconds) == (100_000, 0)

    skipped_form = ExperimentForm("rps", "full_information", ("hedge", "hedge"), 100_000, 3, 2)
    existing = service._spec(skipped_form, 0)
    service.raw_dir.mkdir(parents=True, exist_ok=True)
    (service.raw_dir / f"{existing.run_id}.csv").touch()
    skipped = service.submit_experiment(skipped_form)
    assert (skipped.total, skipped.rounds_total) == (1, 100_000)
    assert wait_for_job(service, skipped.id) == "succeeded"
    assert callbacks == [False, True, True]


def test_horizon_sweeps_expand_to_ordinary_tasks_with_shared_seed_and_replicates(tmp_path, monkeypatch) -> None:
    service = create_service(tmp_path)
    captured = []
    monkeypatch.setattr("web.services.ROUND_PROGRESS_THRESHOLD", 1)

    def record_tasks(function, tasks, *, completed, **kwargs):
        captured.append((function, tasks))
        for _ in tasks:
            completed()
        return []

    monkeypatch.setattr("web.services.run_replicates", record_tasks)
    fixed = ExperimentForm("rps", "full_information", ("hedge", "hedge"), 2, 17, 2, (2, 5))
    existing = service._spec(fixed, 0, 2)
    service.raw_dir.mkdir(parents=True, exist_ok=True)
    (service.raw_dir / f"{existing.run_id}.csv").touch()
    fixed_job = service.submit_experiment(fixed)
    assert wait_for_job(service, fixed_job.id) == "succeeded"

    one_player = AdversarialExperimentForm(
        "historical_frequency_v3", "full_information", "hedge", (2, 4), 3, 17, 2, (3, 6))
    one_player_job = service.submit_adversarial_experiment(one_player)
    assert wait_for_job(service, one_player_job.id) == "succeeded"

    fixed_tasks, one_player_tasks = captured[0][1], captured[1][1]
    assert [(task["horizon"], task["replicate"]) for task in fixed_tasks] == [(2, 1), (5, 0), (5, 1)]
    assert {(task["horizon"], task["n_actions"], task["replicate"]) for task in one_player_tasks} == {
        (horizon, actions, replicate) for horizon in (3, 6) for actions in (2, 4) for replicate in (0, 1)}
    assert {task["seed"] for _, tasks in captured for task in tasks} == {17}
    assert (fixed_job.total, fixed_job.rounds_total) == (3, 12)
    assert (one_player_job.total, one_player_job.rounds_total) == (8, 36)


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


def test_clear_generated_figures_preserves_raw_results_and_custom_games(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    definition = service.create_custom_game("keep me", 2, [2, 2], 0)
    custom_path = service.game_catalog.custom_path(definition.id)
    fixed = run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=2, seed=11,
        output_dir=service.raw_dir, feedback_mode="full_information")
    one_player = run_adversarial_experiment("hedge", horizon=2, seed=12,
        output_dir=service.adversarial_raw_dir)
    preserved = {path: path.read_bytes() for path in (fixed, one_player, custom_path)}
    artifacts = (
        service.figure_dir / "generated.png",
        service.figure_dir / "generated.pdf",
        service.detail_figure_dir / "detail.png",
        service.adversarial_dir / "figures" / "one-player.png",
        service.adversarial_dir / "cache" / "one-player.json",
        service.results_dir / "cache" / "figure_builder" / "collection.pdf",
        service.results_dir / "cache" / "equilibrium_distance" / "distance.json",
    )
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")

    generation = service._detail_figure_generation
    service.clear_generated_figures()

    assert {path: path.read_bytes() for path in preserved} == preserved
    assert not any(path.exists() for path in artifacts)
    assert service._detail_figure_generation == generation + 1
    assert len(service.result_snapshot().records) == 1
    assert len(service.result_snapshot("adversarial").records) == 1


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_clear_generated_figures_route_requires_csrf_and_preserves_mode(tmp_path: Path, mode: str) -> None:
    app, service = create_test_app(tmp_path)
    artifact = service.figure_dir / "generated.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"generated")
    client = app.test_client()
    location = "/" if mode == "fixed" else "/?mode=adversarial"
    page = client.get(location).get_data(as_text=True)
    form = page.split('action="/clear-generated-figures"', 1)[1].split("</form>", 1)[0]
    assert 'name="_csrf_token"' in form
    assert 'name="return_to" value="' + mode + '"' in form
    assert "data-busy-control" in form and "disabled" not in form
    assert "Raw experiment CSVs will be preserved" in form
    assert page.index(">Clear generated figures</button>") < page.index(">Reset all experiment results</button>")

    fields = {"return_to": mode, "confirmation": "clear-generated-figures"}
    assert client.post("/clear-generated-figures", data=fields).status_code == 400
    assert artifact.is_file()
    token = csrf_token(client)
    missing_confirmation = client.post("/clear-generated-figures", data={
        "_csrf_token": token, "return_to": mode,
    })
    assert missing_confirmation.status_code == 302
    assert missing_confirmation.headers["Location"] == location and artifact.is_file()

    response = client.post("/clear-generated-figures", data=fields | {"_csrf_token": token})
    assert response.status_code == 302 and response.headers["Location"] == location
    assert not artifact.exists()
    assert "Cleared generated figures and caches. Raw experiment results were preserved." in (
        client.get(location).get_data(as_text=True))


def test_clear_generated_figures_rejects_busy_jobs(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    artifact = service.figure_dir / "generated.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"generated")
    client = app.test_client()
    token = csrf_token(client)
    blocker, release = block_job_queue(service.jobs)
    try:
        page = client.get("/").get_data(as_text=True)
        form = page.split('action="/clear-generated-figures"', 1)[1].split("</form>", 1)[0]
        assert "data-busy-control disabled" in form
        with pytest.raises(ServiceBusyError):
            service.clear_generated_figures()
        response = client.post("/clear-generated-figures", data={
            "_csrf_token": token, "return_to": "fixed", "confirmation": "clear-generated-figures",
        })
        assert response.status_code == 302 and artifact.is_file()
        assert "wait for the active operation to finish" in client.get("/").get_data(as_text=True)
    finally:
        release.set()
    assert wait_for_job(service, blocker.id) == "succeeded"


def test_delete_fixed_group_removes_all_replicates_and_clears_only_derived_artifacts(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    custom_game = service.create_custom_game("temporary experiment", 2, [2, 2], 7)
    custom_path = service.game_catalog.custom_path(custom_game.id)
    target_paths = [run_cross_play_experiment(custom_game.id, ["hedge", "hedge"], horizon=3, seed=11,
        replicate=replicate, output_dir=service.raw_dir, custom_game_dir=service.game_catalog.custom_game_dir,
        feedback_mode="full_information") for replicate in (0, 1)]
    duplicate = service.raw_dir / "duplicate.csv"
    shutil.copyfile(target_paths[0], duplicate)
    unrelated_fixed = run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=3, seed=12,
        output_dir=service.raw_dir, feedback_mode="full_information")
    unrelated_adversarial = run_adversarial_experiment("hedge", horizon=3, seed=13,
        output_dir=service.adversarial_raw_dir)
    summaries = [row for row in service.result_snapshot().summaries(grouped=True) if row["game"] == custom_game.id]
    assert [row["player"] for row in summaries] == [0, 1]
    assert len({row["group_id"] for row in summaries}) == 1
    group_id = summaries[0]["group_id"]
    assert set(service.result_snapshot().detail_paths(group_id)) == {*target_paths, duplicate}

    artifacts = (
        service.figure_dir / "generated.png",
        service.adversarial_dir / "cache" / "rows.json",
        service.results_dir / "cache" / "figure_builder" / "figure.pdf",
        service.results_dir / "cache" / "equilibrium_distance" / "distance.json",
    )
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"derived")

    assert service.delete_result_group("fixed", group_id) == 3

    assert not any(path.exists() for path in (*target_paths, duplicate, *artifacts))
    assert unrelated_fixed.is_file() and unrelated_adversarial.is_file() and custom_path.is_file()
    assert all(row["game"] != custom_game.id for row in service.result_snapshot().summaries(grouped=True))


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_delete_result_group_route_removes_one_visible_group_and_redirects(tmp_path: Path, kind: str) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    if kind == "fixed":
        target_paths = [run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=3, seed=21,
            replicate=replicate, output_dir=service.raw_dir, feedback_mode="full_information") for replicate in (0, 1)]
        unrelated = run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=3, seed=22,
            output_dir=service.raw_dir, feedback_mode="full_information")
        group_id = next(row["group_id"] for row in service.result_snapshot().summaries(grouped=True) if row["seed"] == 21)
        location = "/"
    else:
        target_paths = [run_adversarial_experiment("hedge", horizon=3, seed=21, replicate=replicate,
            output_dir=service.adversarial_raw_dir) for replicate in (0, 1)]
        unrelated = run_adversarial_experiment("hedge", horizon=3, seed=22,
            output_dir=service.adversarial_raw_dir)
        group_id = next(row["group_id"] for row in service.result_snapshot("adversarial").summaries(grouped=True)
                        if row["base_learner_seed"] == 21)
        location = "/?mode=adversarial"

    page = client.get(browse_url(service, kind, group_id=group_id)).get_data(as_text=True)
    action = f'/experiment-groups/{kind}/{group_id}/delete'
    expected_rows = 2 if kind == "fixed" else 1
    expected_delete_controls = 2 if kind == "fixed" else 1
    assert page.count(f'action="{action}"') == expected_rows
    assert page.count(">Delete experiment</button>") == expected_delete_controls
    confirmation = "Delete this experiment and all of its replicates"
    expected_confirmation = f"{confirmation} for all players?" if kind == "fixed" else f"{confirmation}?"
    assert expected_confirmation in page
    assert page.count('<th class="sticky-actions">Actions</th>') == 1
    assert page.count('<td class="sticky-actions">') == (2 if kind == "fixed" else 1)
    assert f'action="/experiment-groups/{kind}/delete-filtered"' in page
    response = client.post(action, data={"_csrf_token": csrf_token(client)})
    assert response.status_code == 302 and response.headers["Location"] == location
    assert not any(path.exists() for path in target_paths)
    assert unrelated.is_file()


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_bulk_group_deletion_is_atomic_deduplicated_and_cleans_once(tmp_path: Path, monkeypatch, kind: str) -> None:
    service = create_service(tmp_path)
    if kind == "fixed":
        run = lambda seed, replicate: run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=2,
            seed=seed, replicate=replicate, output_dir=service.raw_dir, feedback_mode="full_information")
        snapshot = lambda: service.result_snapshot()
        seed_field = "seed"
    else:
        run = lambda seed, replicate: run_adversarial_experiment("hedge", horizon=2, seed=seed,
            replicate=replicate, output_dir=service.adversarial_raw_dir)
        snapshot = lambda: service.result_snapshot("adversarial")
        seed_field = "base_learner_seed"
    selected = {seed: [run(seed, replicate) for replicate in (0, 1)] for seed in (31, 32)}
    unrelated = run(33, 0)
    groups = {row[seed_field]: row["group_id"] for row in snapshot().summaries(grouped=True)}
    artifact = service.results_dir / "cache" / "figure_builder" / "generated.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"derived")
    cleanup = service._clear_derived_artifacts
    cleanup_calls = 0

    def counted_cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup()

    monkeypatch.setattr(service, "_clear_derived_artifacts", counted_cleanup)
    with pytest.raises(KeyError):
        service.delete_result_groups(kind, [groups[31], "0" * 16])
    assert all(path.is_file() for paths in selected.values() for path in paths)
    assert cleanup_calls == 0

    assert service.delete_result_groups(kind, [groups[31], groups[31], groups[32]]) == 2
    assert not any(path.exists() for paths in selected.values() for path in paths)
    assert unrelated.is_file() and not artifact.exists() and cleanup_calls == 1


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_filtered_deletion_route_rejects_legacy_client_group_ids(tmp_path: Path, monkeypatch, kind: str) -> None:
    app, service = create_test_app(tmp_path)
    monkeypatch.setattr(service, "delete_result_groups",
                        lambda *_: pytest.fail("legacy client group IDs must not be used"))
    client = app.test_client()
    response = client.post(f"/experiment-groups/{kind}/delete-filtered", data={
        "_csrf_token": csrf_token(client), "group_id": ["a" * 16, "b" * 16],
    })
    assert response.status_code == 400
    assert "membership digest" in response.get_json()["error"]


def test_delete_result_group_rejects_busy_and_invalid_requests(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    target = run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=3,
        output_dir=service.raw_dir, feedback_mode="full_information")
    group_id = service.result_snapshot().records[0].group_id
    outside = tmp_path / "outside.csv"
    outside.write_text("preserve", encoding="utf-8")
    blocker, release = block_job_queue(service.jobs)
    client = app.test_client()
    page = client.get(browse_url(service)).get_data(as_text=True)
    action = f'/experiment-groups/fixed/{group_id}/delete'
    form = page.split(f'action="{action}"', 1)[1].split("</form>", 1)[0]
    assert "data-busy-control" in form and "disabled" in form
    response = client.post(action, data={"_csrf_token": csrf_token(client)})
    assert response.status_code == 302 and target.is_file()
    with pytest.raises(ServiceBusyError):
        service.delete_result_groups("fixed", [group_id])
    release.set()
    assert wait_for_job(service, blocker.id) == "succeeded"

    with pytest.raises(ValueError, match="kind"):
        service.delete_result_group("unknown", group_id)
    with pytest.raises(ValueError, match="group"):
        service.delete_result_group("fixed", "../../outside")
    with pytest.raises(KeyError):
        service.delete_result_group("fixed", "0" * 16)
    assert target.is_file() and outside.read_text(encoding="utf-8") == "preserve"


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
    pdf = PdfReader(first.with_suffix(".pdf"))
    information = pdf.pages[0].extract_text()
    assert len(pdf.pages) == 2
    assert "Figure:" in information and "Joint-action distribution" in information
    assert ("Aggregation:  Replicate mean" in information) is grouped


def test_group_detail_figures_use_canonical_replicates_but_delete_all_physical_files(tmp_path, monkeypatch):
    from experiments.plots import plot_equilibrium_convergence, plot_joint_actions

    service = create_service(tmp_path)
    canonical_paths = [run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=2, replicate=replicate,
        output_dir=service.raw_dir, feedback_mode="full_information",
    ) for replicate in (0, 1)]
    group_id = service.result_snapshot().groups("dashboard")[0].records[0].group_id
    cache_stem = service._group_cache_stem(group_id, canonical_paths)
    expected_joint_figure = service.detail_figure_dir / f"{cache_stem}_replicate_mean_joint_actions_blue_lower_origin.png"
    initial_equilibrium_paths, expected_equilibrium_figure, initial_cache_stem = service._group_convergence_figure_path(group_id)
    assert initial_equilibrium_paths == canonical_paths
    assert initial_cache_stem == cache_stem
    joint_inputs = []
    equilibrium_inputs = []
    information_rows = []

    def capture_joint(input_paths, output_path, *args, **kwargs):
        joint_inputs.append(list(input_paths))
        information_rows.append(kwargs["information_rows"])
        write_figure_pair(output_path, b"joint")

    def capture_equilibrium(input_paths, output_path, *args, **kwargs):
        equilibrium_inputs.append(list(input_paths))
        information_rows.append(kwargs["information_rows"])
        write_figure_pair(output_path, b"equilibrium")

    monkeypatch.setattr(plot_joint_actions, "plot_joint_actions", capture_joint)
    monkeypatch.setattr(plot_equilibrium_convergence, "plot_result_equilibrium_distance", capture_equilibrium)
    duplicate = service.raw_dir / "zz_duplicate.csv"
    shutil.copyfile(canonical_paths[0], duplicate)
    snapshot = service.result_snapshot()
    summary = snapshot.summaries(grouped=True)[0]
    assert summary["replicate_count"] == 2
    assert [run["experiment"] for run in summary["runs"]] == [path.name for path in canonical_paths]
    assert snapshot.canonical_detail_paths(group_id) == canonical_paths
    assert set(snapshot.detail_paths(group_id)) == {*canonical_paths, duplicate}

    assert service._group_convergence_figure_path(group_id) == (
        canonical_paths, expected_equilibrium_figure, cache_stem,
    )
    joint_figure = service.group_joint_action_figure(group_id)
    equilibrium_figure, error = wait_for_equilibrium_figure(
        lambda: service.request_group_equilibrium_convergence_figure(group_id)
    )
    assert error is None
    assert joint_figure == expected_joint_figure
    assert equilibrium_figure == expected_equilibrium_figure
    assert joint_inputs == [canonical_paths]
    assert equilibrium_inputs == [canonical_paths]
    assert all(("Replicates", "2") in rows for rows in information_rows)

    assert service.group_joint_action_figure(group_id) == joint_figure
    assert wait_for_equilibrium_figure(
        lambda: service.request_group_equilibrium_convergence_figure(group_id)
    ) == (equilibrium_figure, None)
    assert len(joint_inputs) == len(equilibrium_inputs) == 1

    assert service.delete_result_group("fixed", group_id) == 3
    assert not any(path.exists() for path in (*canonical_paths, duplicate))


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


@pytest.mark.parametrize("clear_method", ["clear_results", "clear_generated_figures"])
@pytest.mark.parametrize("grouped", [False, True])
def test_equilibrium_distance_reuses_paired_cache_and_clears_it(tmp_path, monkeypatch, grouped, clear_method):
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
    getattr(service, clear_method)()
    assert not first.exists() and not first.with_suffix(".pdf").exists()
    assert all(path.exists() for path in paths) == (clear_method == "clear_generated_figures")


def test_dashboard_keeps_active_jobs_outside_display_limit(tmp_path, monkeypatch):
    app, service = create_test_app(tmp_path)
    jobs = [Job(str(i), "Finished", "succeeded", "Done", "now") for i in range(5)]
    jobs += [Job("active", "Active", "running", "Running", "now"), Job("old", "Old", "failed", "Failed", "now")]
    monkeypatch.setattr(service.jobs, "recent", lambda: jobs)
    response = app.test_client().get("/")
    page = response.get_data(as_text=True)
    for job in jobs:
        assert (f'data-job-id="{job.id}"' in page) == (job.id != "old")


def _f3_builder_selection(service, horizon: int):
    context = next(item for item in service.figure_builder.catalog("fixed")["contexts"]
                   if item["scope"] == "rps" and item["player"] == 0)
    profile = context["profiles"][0]["id"]
    selection = parse_profile_selection({
        "mode": "fixed", "context_id": context["id"], "comparison_mode": "regrets",
        "metric": "all", "view": "average", "profiles": [profile],
        "horizon": str(horizon),
    })
    return context, profile, selection


@pytest.mark.parametrize("cleanup", [
    "clear_generated_figures", "clear_results", "delete_result_group",
    "delete_filtered_result_groups",
])
def test_old_figure_builder_publication_is_invalidated_by_shared_cleanup(tmp_path, monkeypatch, cleanup):
    service = create_service(tmp_path)
    raw = run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    context, profile, selection = _f3_builder_selection(service, 3)
    group_id = service.result_snapshot().records[0].group_id
    entered, release = Event(), Event()
    original_publish = service._publish_derived_artifact

    def pause_before_publish(generation, publish):
        entered.set()
        assert release.wait(timeout=20)
        original_publish(generation, publish)

    monkeypatch.setattr(service, "_publish_derived_artifact", pause_before_publish)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="builder") as executor:
        old_work = executor.submit(service.figure_builder.build_collection, selection)
        try:
            assert entered.wait(timeout=15)
            generation = service._detail_figure_generation
            if cleanup == "clear_generated_figures":
                service.clear_generated_figures()
            elif cleanup == "clear_results":
                service.clear_results()
            elif cleanup == "delete_result_group":
                assert service.delete_result_group("fixed", group_id) == 1
            else:
                values = {
                    "mode": "fixed", "scope": "rps", "context": context["id"],
                    "comparison_mode": "profiles", "feedback": "full_information",
                    "horizon": "3", "profiles": [profile], "player": "0",
                }
                preview = service.preview_filtered_result_groups("fixed", values)
                assert preview.count == 1
                assert service.delete_filtered_result_groups(
                    "fixed", values, preview.membership_digest) == 1
            assert service._detail_figure_generation == generation + 1
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="derived artifact generation was invalidated"):
            old_work.result(timeout=15)

    assert not list(service.figure_builder.output_dir.glob("*"))
    assert not list(tmp_path.glob(".selection-*"))
    assert raw.exists() == (cleanup == "clear_generated_figures")


def test_builder_and_distance_cache_old_work_cannot_republish_but_fresh_work_can(tmp_path, monkeypatch):
    service = create_service(tmp_path)
    raw = run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    _, _, selection = _f3_builder_selection(service, 3)
    entered = {"builder": Event(), "cache": Event()}
    release = Event()
    original_publish = service._publish_derived_artifact

    def pause_before_publish(generation, publish):
        kind = "cache" if current_thread().name.startswith("equilibrium-distance") else "builder"
        entered[kind].set()
        assert release.wait(timeout=20)
        original_publish(generation, publish)

    monkeypatch.setattr(service, "_publish_derived_artifact", pause_before_publish)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="builder") as executor:
        old_builder = executor.submit(service.figure_builder.build_collection, selection)
        try:
            assert service.request_equilibrium_convergence_figure(raw.name) == (None, None)
            old_distance = next(iter(service._convergence_futures.values()))
            assert all(event.wait(timeout=15) for event in entered.values())
            service.clear_generated_figures()
            assert service._convergence_futures == {}
        finally:
            release.set()
        for old_work in (old_builder, old_distance):
            with pytest.raises(RuntimeError, match="derived artifact generation was invalidated"):
                old_work.result(timeout=15)

    assert raw.is_file()
    assert not list((service.results_dir / "cache").rglob("*.json"))
    assert not list(service.figure_builder.output_dir.glob("*"))
    assert not list(service.detail_figure_dir.glob("*"))
    assert not list(tmp_path.glob(".selection-*"))
    assert not list(tmp_path.glob(".equilibrium-convergence-*"))

    monkeypatch.setattr(service, "_publish_derived_artifact", original_publish)
    fresh = service.figure_builder.build_collection(selection)
    assert len(fresh["figures"]) == 2
    assert all((service.figure_builder.output_dir / item["filename"]).is_file()
               and (service.figure_builder.output_dir / item["pdf_filename"]).is_file()
               for item in fresh["figures"])
    figure, error = wait_for_equilibrium_figure(
        lambda: service.request_equilibrium_convergence_figure(raw.name))
    assert error is None and figure is not None and figure.is_file()
    assert figure.with_suffix(".pdf").is_file()
    assert len(list((service.results_dir / "cache" / "equilibrium_distance").glob("*.json"))) == 1


def test_existing_detail_figure_invalidation_rejects_old_render_and_allows_new(tmp_path, monkeypatch):
    from experiments.plots import plot_equilibrium_convergence as plotting

    service = create_service(tmp_path)
    raw = run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    entered, release = Event(), Event()

    def render(input_paths, output_path, **kwargs):
        entered.set()
        assert release.wait(timeout=20)
        write_figure_pair(output_path, b"distance")

    monkeypatch.setattr(plotting, "plot_result_equilibrium_distance", render)
    assert service.request_equilibrium_convergence_figure(raw.name) == (None, None)
    old_work = next(iter(service._convergence_futures.values()))
    try:
        assert entered.wait(timeout=15)
        service.clear_generated_figures()
        assert service._convergence_futures == {}
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="equilibrium convergence figure generation was invalidated"):
        old_work.result(timeout=15)
    assert not list(service.detail_figure_dir.glob("*"))
    assert not list(tmp_path.glob(".equilibrium-convergence-*"))

    figure, error = wait_for_equilibrium_figure(
        lambda: service.request_equilibrium_convergence_figure(raw.name))
    assert error is None and figure is not None and figure.is_file()
    assert figure.with_suffix(".pdf").is_file()


def test_joint_action_request_started_before_cleanup_cannot_publish_afterward(tmp_path, monkeypatch):
    import web.services as services_module

    service = create_service(tmp_path)
    raw = run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    entered, release = Event(), Event()
    original_rows = services_module.iter_result_rows

    def pause_before_detail_lock(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=20)
        return original_rows(*args, **kwargs)

    monkeypatch.setattr(services_module, "iter_result_rows", pause_before_detail_lock)
    with ThreadPoolExecutor(max_workers=1) as executor:
        old_work = executor.submit(service.joint_action_figure, raw.name)
        try:
            assert entered.wait(timeout=15)
            service.clear_generated_figures()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="joint-action figure generation was invalidated"):
            old_work.result(timeout=15)

    assert not list(service.detail_figure_dir.glob("*"))
    monkeypatch.setattr(services_module, "iter_result_rows", original_rows)
    fresh = service.joint_action_figure(raw.name)
    assert fresh.is_file() and fresh.with_suffix(".pdf").is_file()
