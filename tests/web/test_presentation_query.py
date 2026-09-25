"""Parity coverage for the Phase 3A projector used by dashboard browsing."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.result_catalog import ResultSet
from experiments.scenarios.adversarial import run_adversarial_experiment
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.web.support import create_test_app, run_node
from web.browsing import query_string
from web.presentation_query import parse_dashboard_query, project_dashboard_query


@pytest.fixture
def cases(tmp_path: Path):
    app, service = create_test_app(tmp_path)
    fixed = [
        ("rps", ("hedge", "hedge"), "full_information", 2),
        ("rps", ("hedge", "hedge"), "full_information", 3),
        ("rps", ("ito_hedge", "ito_hedge"), "full_information", 2),
        ("rps", ("ito_hedge", "ito_hedge"), "full_information", 3),
        ("rps", ("auer_exp3", "bm_exp3"), "bandit", 2),
        ("rpsls", ("hedge", "hedge"), "full_information", 2),
    ]
    for game, profile, feedback, horizon in fixed:
        run_cross_play_experiment(
            game, profile, feedback_mode=feedback, horizon=horizon, seed=42,
            output_dir=service.raw_dir,
        )
    custom = service.create_custom_game("Asymmetric three", 3, [2, 2, 2], 7)
    run_cross_play_experiment(
        custom.id, ("hedge", "ito_hedge", "hedge"), feedback_mode="full_information",
        horizon=2, seed=42, output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
    )
    adversarial = [
        ("historical_frequency_v3", "hedge", "full_information", 2, 2),
        ("historical_frequency_v3", "hedge", "full_information", 3, 2),
        ("historical_frequency_v3", "hedge", "full_information", 2, 3),
        ("historical_frequency_v3", "exp3_ix", "bandit", 2, 2),
        ("lazy_random_walk_v1", "hedge", "full_information", 2, 2),
    ]
    for environment, profile, feedback, actions, horizon in adversarial:
        run_adversarial_experiment(
            profile, environment=environment, feedback_mode=feedback,
            n_actions=actions, horizon=horizon, seed=42,
            output_dir=service.adversarial_raw_dir,
        )
    return app, service, custom.id


def selection(service, *, mode="fixed", scope="rps", player=0, action=2,
              feedback="full_information", horizon=2, profiles=("hedge_vs_hedge",),
              comparison_mode="profiles", metric="all", view="all", sort=None,
              direction="asc"):
    catalog = service.figure_builder.catalog(mode)
    context = next(item for item in catalog["contexts"]
                   if item["scope"] == scope
                   and (mode == "adversarial" or item["player"] == player or player == "all"))
    values = {
        "mode": mode, "scope": scope, "context": context["id"],
        "comparison_mode": comparison_mode, "feedback": feedback,
        "horizon": str(horizon), "profiles": profiles, "metric": metric,
        "view": view, "sort": sort, "direction": direction,
    }
    values["player" if mode == "fixed" else "action"] = str(
        player if mode == "fixed" else action)
    return parse_dashboard_query(values, catalog), catalog, context


def projection(service, query, catalog):
    return project_dashboard_query(
        service.result_snapshot(query.mode), catalog, query,
        presentations=service.game_presentations if query.mode == "fixed" else {},
    )


DOM_PARITY = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page);
const rows = [...dom.window.document.querySelectorAll(".summary-row")];
assert.deepEqual([...new Set(rows.map(row => row.dataset.resultKey))], payload.groupIds);
assert.deepEqual(rows.map(row => [row.dataset.resultKey, Number(row.dataset.player || 0)]),
    payload.rows);
dom.window.close();
'''


def assert_dom_parity(app, query, context, projected):
    response = app.test_client().get("/?" + query_string(query))
    assert response.status_code == 200
    run_node(DOM_PARITY, payload={
        "page": response.get_data(as_text=True), "groupIds": list(projected.group_ids),
        "rows": [[row.group_id, row.player] for row in projected.rows],
    }, jsdom=True)


@pytest.mark.parametrize("feedback,horizon,profiles,expected", [
    ("full_information", 2, ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge"), 2),
    ("full_information", 3, ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge"), 2),
    ("bandit", 2, ("auer_exp3_vs_bm_exp3",), 1),
    ("both", 2, ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge",
                  "auer_exp3_vs_bm_exp3"), 3),
    ("full_information", 2, (), 0),
])
def test_fixed_filter_membership_matches_current_dom(cases, feedback, horizon, profiles, expected):
    app, service, _ = cases
    query, catalog, context = selection(
        service, feedback=feedback, horizon=horizon, profiles=profiles)
    projected = projection(service, query, catalog)
    assert len(projected.groups) == expected
    assert all(len(group.matching_rows) == 1 and group.matching_rows[0].player == 0
               and len(group.group.records[0].profile) == 2 for group in projected.groups)
    assert_dom_parity(app, query, context, projected)


def test_fixed_scope_player_custom_and_complete_groups(cases):
    app, service, custom = cases
    for scope, player, profile, expected_players in (
        ("rpsls", 1, "hedge_vs_hedge", 2),
        (custom, 2, "hedge_vs_ito_hedge_vs_hedge", 3),
    ):
        query, catalog, context = selection(
            service, scope=scope, player=player, profiles=(profile,))
        projected = projection(service, query, catalog)
        assert len(projected.groups) == 1
        assert len(projected.groups[0].group.records[0].profile) == expected_players
        assert [row.player for row in projected.rows] == [player]
        assert_dom_parity(app, query, context, projected)


@pytest.mark.parametrize("scope,feedback,action,horizon,profiles,expected", [
    ("historical_frequency_v3", "full_information", 2, 2, ("hedge",), 1),
    ("historical_frequency_v3", "full_information", 3, 2, ("hedge",), 1),
    ("historical_frequency_v3", "full_information", 2, 3, ("hedge",), 1),
    ("historical_frequency_v3", "bandit", 2, 2, ("exp3_ix",), 1),
    ("historical_frequency_v3", "both", 2, 2, ("hedge", "exp3_ix"), 2),
    ("lazy_random_walk_v1", "full_information", 2, 2, ("hedge",), 1),
    ("historical_frequency_v3", "full_information", 2, 2, (), 0),
])
def test_one_player_filter_membership_matches_current_dom(
    cases, scope, feedback, action, horizon, profiles, expected,
):
    app, service, _ = cases
    query, catalog, context = selection(
        service, mode="adversarial", scope=scope, action=action,
        feedback=feedback, horizon=horizon, profiles=profiles)
    projected = projection(service, query, catalog)
    assert len(projected.groups) == expected
    assert all(len(group.matching_rows) == 1 for group in projected.groups)
    assert_dom_parity(app, query, context, projected)


@pytest.mark.parametrize("mode,options,expected", [
    ("fixed", {"comparison_mode": "horizons", "horizon": "all",
               "view": "horizon_scaling", "profiles": ("hedge_vs_hedge",)}, 2),
    ("adversarial", {"mode": "adversarial", "scope": "historical_frequency_v3",
                     "comparison_mode": "actions", "action": "all",
                     "horizon": 2, "profiles": ("hedge",)}, 2),
    ("adversarial", {"mode": "adversarial", "scope": "historical_frequency_v3",
                     "comparison_mode": "horizons", "action": 2,
                     "horizon": "all", "view": "horizon_scaling",
                     "profiles": ("hedge",)}, 2),
])
def test_comparison_modes_match_current_dom(cases, mode, options, expected):
    app, service, _ = cases
    query, catalog, context = selection(service, **options)
    projected = projection(service, query, catalog)
    assert query.mode == mode
    assert len(projected.groups) == expected
    assert_dom_parity(app, query, context, projected)


@pytest.mark.parametrize("mode,scope,profile", [
    ("fixed", "rps", "hedge_vs_hedge"),
    ("adversarial", "historical_frequency_v3", "hedge"),
])
def test_retired_view_is_rejected_and_ordinary_regret_views_validate(cases, mode, scope, profile):
    _, service, _ = cases
    options = dict(mode=mode, scope=scope, comparison_mode="regrets", profiles=(profile,))
    with pytest.raises(ValueError, match="invalid regret view"):
        selection(service, view="log_log_fit", **options)
    for view in ("all", "average", "sqrt_scaling"):
        query, catalog, _ = selection(service, view=view, **options)
        projected = projection(service, query, catalog)
        assert len(projected.groups) == 1
        assert {bucket[0] for bucket in projected.global_minima} == {
            f"{kind}_{metric}" for kind in ("average", "sqrt_scaling")
            if view in ("all", kind) for metric in ("external", "internal", "swap")
        }


def test_default_order_and_numeric_sorts_are_deterministic(cases):
    _, service, _ = cases
    query, catalog, _ = selection(
        service, profiles=("hedge_vs_hedge", "ito_hedge_vs_ito_hedge"))
    snapshot = service.result_snapshot()
    expected = [group.records[0].group_id for group in snapshot.groups("dashboard")
                if group.records[0].group_id in
                {group.group_id for group in projection(service, query, catalog).groups}]
    normal = projection(service, query, catalog)
    assert list(normal.group_ids) == expected
    for direction in ("asc", "desc"):
        sorted_query = replace(query, sort="average_external", direction=direction)
        result = projection(service, sorted_query, catalog)
        values = [group.matching_rows[0].display_regrets["average_external"]
                  for group in result.groups]
        assert values == sorted(values, reverse=direction == "desc")
        assert [group.canonical_rank for group in result.groups
                if values[0] == group.matching_rows[0].display_regrets["average_external"]] == sorted(
                    group.canonical_rank for group in result.groups
                    if values[0] == group.matching_rows[0].display_regrets["average_external"])
    query, catalog, _ = selection(
        service, mode="adversarial", scope="historical_frequency_v3",
        action="all", comparison_mode="actions", profiles=("hedge",),
        feedback="full_information", horizon=2)
    result = projection(service, replace(query, sort="action", direction="desc"), catalog)
    assert [row.summary["n_actions"] for row in result.rows] == [3, 2]


def test_ties_global_best_and_hidden_players_are_separate(cases):
    _, service, _ = cases
    query, catalog, _ = selection(
        service, profiles=("hedge_vs_hedge", "ito_hedge_vs_ito_hedge"),
        metric="external", view="average")
    original = service.result_snapshot()
    selected = set(projection(service, query, catalog).group_ids)
    records = []
    for record in original.records:
        values = [dict(player_values) for player_values in record.final_values]
        if record.group_id in selected:
            values[0]["average_external_regret"] = 0.1
            values[1]["average_external_regret"] = -10.0
        records.append(replace(record, final_values=tuple(values)))
    synthetic = ResultSet(tuple(records))
    result = project_dashboard_query(synthetic, catalog, query)
    assert len(result.groups) == 2
    assert len(result.best_cells) == 2
    assert result.best_cells == {
        (group.group_id, 0, "average_external") for group in result.groups}
    assert len(result.global_minima) == 1
    assert next(iter(result.global_minima.values())) == 0.1
    assert all(column != "sqrt_scaling_external" for _, _, column in result.best_cells)
    tied = project_dashboard_query(
        synthetic, catalog, replace(query, sort="average_external", direction="desc"))
    assert [group.canonical_rank for group in tied.groups] == [
        group.canonical_rank for group in result.groups]
    legacy = project_dashboard_query(
        synthetic, catalog, replace(query, player="all", sort="average_external"))
    assert legacy.pageable is False and legacy.sort_applied is False
    assert len(legacy.rows) == 4
    assert all(group.group.records[0].profile for group in legacy.groups)
    assert {key[2] for key in legacy.global_minima} == {"0", "1"}


def test_projection_display_values_match_current_view_model(cases):
    from web.view_models import _display_regrets

    _, service, _ = cases
    for options in (
        {"profiles": ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge")},
        {"mode": "adversarial", "scope": "historical_frequency_v3",
         "profiles": ("hedge",), "action": 2},
    ):
        query, catalog, _ = selection(service, **options)
        projected = projection(service, query, catalog)
        assert projected.rows
        for row in projected.rows:
            assert row.display_regrets == _display_regrets(row.summary)


def test_one_player_regret_sort_and_feedback_highlight_buckets(cases):
    _, service, _ = cases
    query, catalog, _ = selection(
        service, mode="adversarial", scope="historical_frequency_v3",
        feedback="both", action=2, horizon=2, profiles=("hedge", "exp3_ix"),
        metric="external", view="average",
    )
    snapshot = service.result_snapshot("adversarial")
    chosen = set(projection(service, query, catalog).group_ids)
    records = []
    for record in snapshot.records:
        values = [dict(value) for value in record.final_values]
        if record.group_id in chosen:
            values[0]["average_external_regret"] = (
                0.2 if record.feedback_mode == "full_information" else -1.0)
        records.append(replace(record, final_values=tuple(values)))
    synthetic = ResultSet(tuple(records))
    result = project_dashboard_query(synthetic, catalog, query)
    assert len(result.groups) == 2
    assert {key[3] for key in result.global_minima} == {"full_information", "bandit"}
    assert len(result.best_cells) == 2
    sorted_result = project_dashboard_query(
        synthetic, catalog, replace(query, sort="average_external", direction="asc"))
    assert [row.display_regrets["average_external"] for row in sorted_result.rows] == [-1.0, 0.2]


def test_projector_preserves_first_replicate_and_detail_membership(tmp_path):
    app, service = create_test_app(tmp_path)
    paths = [run_cross_play_experiment(
        "rps", ("hedge", "hedge"), feedback_mode="full_information",
        horizon=2, seed=42, replicate=replicate, output_dir=service.raw_dir,
    ) for replicate in (0, 1)]
    duplicate = service.raw_dir / "zz_duplicate.csv"
    duplicate.write_bytes(paths[0].read_bytes())
    query, catalog, _ = selection(service)
    projected = projection(service, query, catalog)
    assert len(projected.groups) == 1
    assert projected.groups[0].group.paths == paths
    assert set(service.result_snapshot().detail_paths(projected.group_ids[0])) == {*paths, duplicate}
    detail = app.test_client().get(
        f"/experiment-groups/fixed/{projected.group_ids[0]}/players/0").get_json()
    assert [run["experiment"] for run in detail["runs"]] == [path.name for path in paths]


def test_empty_and_legacy_all_player_queries_are_explicit(cases):
    _, service, _ = cases
    catalog = service.figure_builder.catalog("fixed")
    empty = parse_dashboard_query({}, catalog)
    assert empty.context_id == "" and empty.profiles == ()
    assert projection(service, empty, catalog).group_ids == ()
    normal, catalog, _ = selection(service)
    values = {
        "mode": "fixed", "scope": normal.scope, "context": normal.context_id,
        "comparison_mode": "profiles", "feedback": normal.feedback,
        "horizon": normal.horizon, "profiles": normal.profiles, "player": "all",
    }
    legacy = parse_dashboard_query(values, catalog)
    projected = projection(service, replace(legacy, sort="average_external"), catalog)
    assert projected.pageable is False and projected.sort_applied is False
    assert len(projected.rows) == 2
    with pytest.raises(ValueError, match="action is unavailable"):
        query, catalog, context = selection(
            service, mode="adversarial", scope="historical_frequency_v3",
            action=2, profiles=("hedge",))
        parse_dashboard_query({
            "mode": "adversarial", "scope": query.scope, "context": context["id"],
            "comparison_mode": "profiles", "feedback": query.feedback,
            "horizon": query.horizon, "profiles": query.profiles, "action": "99",
        }, catalog)


@pytest.mark.parametrize("mode,options", [
    ("fixed", {}),
    ("adversarial", {"mode": "adversarial", "scope": "historical_frequency_v3",
                     "action": "all", "comparison_mode": "actions", "profiles": ("hedge",)}),
])
def test_parser_accepts_current_builder_saved_state_without_trusting_result_keys(
    cases, mode, options,
):
    _, service, _ = cases
    query, catalog, _ = selection(service, **options)
    saved = {
        "mode": mode, "scope": query.scope, "context": query.context_id,
        "comparisonMode": query.comparison_mode, "feedback": query.feedback,
        "horizon": query.horizon, "profiles": list(query.profiles),
        "player": str(query.player) if mode == "fixed" else "0",
        "action": str(query.action) if mode == "adversarial" else None,
        "metric": query.metric, "view": query.view,
        "resultKeys": ["untrusted-client-key"],
    }
    parsed = parse_dashboard_query(saved, catalog)
    assert parsed == query
    assert projection(service, parsed, catalog).group_ids
    with pytest.raises(ValueError, match="conflicting comparison modes"):
        parse_dashboard_query(saved | {"comparison_mode": "regrets"}, catalog)
    if mode == "adversarial":
        with pytest.raises(ValueError, match="invalid one-player action"):
            parse_dashboard_query(saved | {"player": "1"}, catalog)

@pytest.mark.parametrize("changes", [
    {"mode": "other"}, {"context": "missing"}, {"scope": "rpsls"},
    {"player": "-1"}, {"player": "2"}, {"horizon": "0"}, {"horizon": "4"}, {"feedback": "unknown"},
    {"profiles": ("missing",)}, {"metric": "unknown"}, {"view": "unknown"},
    {"sort": "filesystem_path"}, {"action": "2"},
])
def test_invalid_query_never_broadens_context(cases, changes):
    _, service, _ = cases
    good, catalog, _ = selection(service)
    values = {
        "mode": good.mode, "scope": good.scope, "context": good.context_id,
        "comparison_mode": good.comparison_mode, "feedback": good.feedback,
        "horizon": good.horizon, "profiles": good.profiles,
        "metric": good.metric, "view": good.view, "player": str(good.player),
    } | changes
    with pytest.raises(ValueError):
        parse_dashboard_query(values, catalog)
    assert project_dashboard_query(service.result_snapshot(), catalog,
                                   parse_dashboard_query({}, catalog)).group_ids == ()


def test_builder_detail_and_live_pages_are_unchanged(cases):
    app, service, _ = cases
    client = app.test_client()
    before = {mode: (
        client.get("/", query_string={"mode": mode}).data,
        client.get("/figure-builder/options", query_string={"mode": mode}).data,
    ) for mode in ("fixed", "adversarial")}
    for mode in ("fixed", "adversarial"):
        catalog = service.figure_builder.catalog(mode)
        for context in catalog["contexts"]:
            selected = context["profiles"][0]
            horizon = (selected["horizons"][0] if mode == "fixed"
                       else selected["availability"][str(selected["actions"][0])][0])
            values = {
                "mode": mode, "scope": context["scope"], "context": context["id"],
                "comparison_mode": "profiles", "feedback": selected["feedback_mode"],
                "horizon": str(horizon), "profiles": [selected["id"]],
                "player" if mode == "fixed" else "action":
                    str(context["player"] if mode == "fixed" else selected["actions"][0]),
            }
            query = parse_dashboard_query(values, catalog)
            project_dashboard_query(service.result_snapshot(mode), catalog, query,
                                    presentations=service.game_presentations)
        page, options = before[mode]
        assert client.get("/", query_string={"mode": mode}).data == page
        current_options = client.get("/figure-builder/options", query_string={"mode": mode}).data
        assert current_options == options
        assert sha256(current_options).digest() == sha256(options).digest()
    query, catalog, _ = selection(service)
    result = projection(service, query, catalog)
    group_id = result.group_ids[0]
    runs_before = service.result_snapshot().groups("dashboard")
    group = next(group for group in runs_before if group.records[0].group_id == group_id)
    expected = [path.name for path in group.paths]
    detail = client.get(f"/experiment-groups/fixed/{group_id}/players/0")
    assert detail.status_code == 200
    assert [run["experiment"] for run in detail.get_json()["runs"]] == expected
