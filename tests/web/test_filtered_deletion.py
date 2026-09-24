"""Server-authoritative filtered-deletion preview and confirmation tests."""

from pathlib import Path

import pytest

from experiments.scenarios.adversarial import run_adversarial_experiment
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.web.support import create_test_app, csrf_token
from web.presentation_query import parse_dashboard_query, project_dashboard_query


def record(service, kind, *, horizon=2, seed=42, action=2, replicate=0, algorithm="hedge"):
    if kind == "fixed":
        return run_cross_play_experiment(
            "rps", (algorithm, algorithm), feedback_mode="full_information",
            horizon=horizon, seed=seed, replicate=replicate, output_dir=service.raw_dir,
        )
    return run_adversarial_experiment(
        algorithm, environment="historical_frequency_v3",
        feedback_mode="full_information", n_actions=action, horizon=horizon,
        seed=seed, replicate=replicate, output_dir=service.adversarial_raw_dir,
    )


def filter_state(service, kind, *, comparison="profiles", horizon="2", action="2",
                 profiles=None, scope=None, player="0"):
    scope = scope or ("rps" if kind == "fixed" else "historical_frequency_v3")
    catalog = service.figure_builder.catalog(kind)
    context = next(item for item in catalog["contexts"]
                   if item["scope"] == scope and (kind == "adversarial" or item["player"] == int(player)))
    return {
        "mode": kind, "scope": scope, "context": context["id"],
        "comparisonMode": comparison, "feedback": "full_information",
        "horizon": str(horizon),
        "profiles": profiles if profiles is not None
            else (["hedge_vs_hedge"] if kind == "fixed" else ["hedge"]),
        "player": str(player) if kind == "fixed" else "0",
        "action": str(action) if kind == "adversarial" else None,
        "metric": "all", "view": "horizon_scaling" if comparison == "horizons" else "all",
    }


def request_preview(client, kind, state, token):
    return client.post(
        f"/experiment-groups/{kind}/delete-filtered/preview",
        data={"_csrf_token": token, **state}, headers={"Accept": "application/json"},
    )


def request_delete(client, kind, state, token, digest):
    return client.post(
        f"/experiment-groups/{kind}/delete-filtered",
        data={"_csrf_token": token, **state, "digest": digest},
        headers={"Accept": "application/json"},
    )


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_preview_uses_complete_projection_and_ignores_presentation_only_fields(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    record(service, kind)
    if kind == "fixed":
        record(service, kind, algorithm="ito_hedge")
        profiles = ["hedge_vs_hedge", "ito_hedge_vs_ito_hedge"]
        comparison, action = "profiles", "2"
    else:
        record(service, kind, action=3)
        profiles = ["hedge"]
        comparison, action = "actions", "all"
    state = filter_state(service, kind, comparison=comparison, action=action, profiles=profiles)
    client = app.test_client()
    token = csrf_token(client)
    options_before = client.get("/figure-builder/options", query_string={"mode": kind}).data
    catalog = service.figure_builder.catalog(kind)
    query = parse_dashboard_query(state, catalog)
    projected = project_dashboard_query(service.result_snapshot(kind), catalog, query)
    response = request_preview(client, kind, state, token)
    assert response.status_code == 200
    assert service.figure_builder.catalog(kind, service.result_snapshot(kind)) == catalog
    preview = response.get_json()
    assert preview["count"] == len(projected.group_ids) == 2
    assert len(preview["digest"]) == 64
    assert "group_ids" not in preview and "paths" not in preview
    changed_display = state | {
        "sort": "horizon", "direction": "desc", "page": "999",
        "page_size": "1", "metric": "not-a-metric", "view": "not-a-view",
        "group_id": ["f" * 16],
    }
    assert request_preview(client, kind, changed_display, token).get_json() == preview
    assert client.get("/figure-builder/options", query_string={"mode": kind}).data == options_before


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_zero_matches_preview_and_confirmation_do_no_work(tmp_path: Path, monkeypatch, kind: str):
    app, service = create_test_app(tmp_path)
    path = record(service, kind)
    state = filter_state(service, kind) | {"profiles": []}
    monkeypatch.setattr(service, "_clear_derived_artifacts",
                        lambda: pytest.fail("zero-match deletion must not clean artifacts"))
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    assert preview["count"] == 0
    response = request_delete(client, kind, state, token, preview["digest"])
    assert response.status_code == 200 and response.get_json()["deleted"] == 0
    assert path.is_file()


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_confirm_deletes_only_previewed_groups_despite_nonmatching_addition(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    selected = record(service, kind)
    state = filter_state(service, kind)
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    unrelated = record(service, kind, seed=99)
    response = request_delete(client, kind, state | {
        "group_id": ["f" * 16], "filename": "../../untrusted.csv",
    }, token, preview["digest"])
    assert response.status_code == 200 and response.get_json()["deleted"] == 1
    assert not selected.exists() and unrelated.is_file()


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_new_matching_group_after_preview_aborts_every_delete(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    paths = [record(service, kind, horizon=horizon) for horizon in (2, 3)]
    state = filter_state(service, kind, comparison="horizons", horizon="all")
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    assert preview["count"] == 2
    paths.append(record(service, kind, horizon=4))
    response = request_delete(client, kind, state, token, preview["digest"])
    assert response.status_code == 409
    assert response.get_json()["reconfirmation_required"] is True
    assert response.get_json()["count"] == 3
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_matching_group_disappearing_after_preview_aborts(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    surviving = record(service, kind, horizon=2)
    record(service, kind, horizon=3)
    state = filter_state(service, kind, comparison="horizons", horizon="all")
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    group = next(row["group_id"] for row in service.result_snapshot(kind).summaries(grouped=True)
                 if row["horizon"] == 3)
    service.delete_result_group(kind, group)
    response = request_delete(client, kind, state, token, preview["digest"])
    assert response.status_code == 409 and response.get_json()["reconfirmation_required"] is True
    assert surviving.is_file()


def test_fixed_duplicate_file_added_after_preview_uses_authoritative_detail_paths(tmp_path: Path):
    kind = "fixed"
    app, service = create_test_app(tmp_path)
    paths = [record(service, kind, replicate=replicate) for replicate in (0, 1)]
    state = filter_state(service, kind)
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    group = service.result_snapshot(kind).groups("dashboard")[0]
    assert len(group.paths) == 2
    duplicate = paths[0].parent / "zz_duplicate.csv"
    duplicate.write_bytes(paths[0].read_bytes())
    assert set(service.result_snapshot(kind).detail_paths(group.records[0].group_id)) == {*paths, duplicate}
    detail = client.get(f"/experiment-groups/fixed/{group.records[0].group_id}/players/0")
    assert detail.status_code == 200 and len(detail.get_json()["runs"]) == 2
    response = request_delete(client, kind, state, token, preview["digest"])
    assert response.status_code == 200 and response.get_json()["deleted"] == 1
    assert not any(path.exists() for path in (*paths, duplicate))


def test_one_player_duplicate_after_preview_requires_reconfirmation_but_single_delete_still_works(
    tmp_path: Path,
):
    app, service = create_test_app(tmp_path)
    paths = [record(service, "adversarial", replicate=replicate) for replicate in (0, 1)]
    state = filter_state(service, "adversarial")
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, "adversarial", state, token).get_json()
    group_id = service.result_snapshot("adversarial").records[0].group_id
    duplicate = paths[0].parent / "zz_duplicate.csv"
    duplicate.write_bytes(paths[0].read_bytes())
    response = request_delete(client, "adversarial", state, token, preview["digest"])
    assert response.status_code == 409 and response.get_json()["reconfirmation_required"] is True
    assert all(path.is_file() for path in (*paths, duplicate))
    assert service.delete_result_group("adversarial", group_id) == 3
    assert not any(path.exists() for path in (*paths, duplicate))


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_invalid_state_and_digest_cannot_delete(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    path = record(service, kind)
    state = filter_state(service, kind)
    client = app.test_client()
    token = csrf_token(client)
    for invalid in (
        state | {"context": "missing"},
        state | {"mode": "adversarial" if kind == "fixed" else "fixed"},
        state | {"profiles": ["unknown"]},
    ):
        assert request_preview(client, kind, invalid, token).status_code == 400
    preview = request_preview(client, kind, state, token).get_json()
    assert request_delete(client, kind, state, token, "invalid").status_code == 400
    changed = request_delete(client, kind, state, token, "0" * 64)
    assert changed.status_code == 409 and changed.get_json()["reconfirmation_required"] is True
    assert path.is_file()
    other_mode = "regrets"
    tampered = request_delete(client, kind, state | {"comparisonMode": other_mode}, token, preview["digest"])
    assert tampered.status_code == 409 and path.is_file()


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_context_disappearing_entirely_requires_reconfirmation(tmp_path: Path, kind: str):
    app, service = create_test_app(tmp_path)
    record(service, kind)
    state = filter_state(service, kind)
    client = app.test_client()
    token = csrf_token(client)
    preview = request_preview(client, kind, state, token).get_json()
    group_id = service.result_snapshot(kind).records[0].group_id
    service.delete_result_group(kind, group_id)
    response = request_delete(client, kind, state, token, preview["digest"])
    assert response.status_code == 409 and response.get_json()["reconfirmation_required"] is True


def test_custom_asymmetric_group_preview_and_deletion_keep_full_scientific_group(tmp_path: Path):
    app, service = create_test_app(tmp_path)
    custom = service.create_custom_game("Asymmetric three", 3, [2, 2, 2], 7)
    path = run_cross_play_experiment(
        custom.id, ("hedge", "ito_hedge", "hedge"), feedback_mode="full_information",
        horizon=2, seed=42, output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
    )
    state = filter_state(
        service, "fixed", scope=custom.id, player="2",
        profiles=["hedge_vs_ito_hedge_vs_hedge"],
    )
    client = app.test_client()
    token = csrf_token(client)
    catalog = service.figure_builder.catalog("fixed")
    projected = project_dashboard_query(service.result_snapshot(), catalog,
                                        parse_dashboard_query(state, catalog))
    assert len(projected.groups) == 1
    assert [row.player for row in projected.rows] == [2]
    group_id = projected.group_ids[0]
    assert len(projected.groups[0].group.records[0].profile) == 3
    assert client.get(f"/experiment-groups/fixed/{group_id}/players/2").status_code == 200
    preview = request_preview(client, "fixed", state, token).get_json()
    assert preview["count"] == 1
    response = request_delete(client, "fixed", state, token, preview["digest"])
    assert response.status_code == 200 and not path.exists()
