"""Phase 3C production browsing, complete-group pagination, and URL coordination."""

from dataclasses import replace
from html import unescape
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlsplit

import pytest
from werkzeug.datastructures import MultiDict

from experiments.result_catalog import ResultSet
from experiments.scenarios.adversarial import run_adversarial_experiment
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.web.support import browse_url, create_test_app, csrf_token, dashboard_data, run_node
from web.browsing import paginate_projection, parse_browsing_query, query_string
from web.presentation_query import project_dashboard_query


def _rows(response):
    return re.findall(rb'<tr\b(?=[^>]*class="summary-row")[^>]*>', response.data)


def _group_ids(response):
    return [value.decode() for value in
            (re.search(rb'data-result-key="([0-9a-f]+)"', row).group(1) for row in _rows(response))]


def _params(url):
    return MultiDict(parse_qsl(urlsplit(url).query))


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    app, service = create_test_app(tmp_path_factory.mktemp("browsing"))
    full = service.algorithms_by_feedback_mode["full_information"]
    bandit = service.algorithms_by_feedback_mode["bandit"]
    profiles = ([(left, right, "full_information")
                 for left in full for right in full][:28]
                + [(left, right, "bandit") for left in bandit for right in bandit])
    assert len(profiles) == 53
    for left, right, feedback in profiles:
        run_cross_play_experiment(
            "rps", (left, right), feedback_mode=feedback,
            horizon=2, seed=7, output_dir=service.raw_dir,
        )
    for actions in range(2, 29):
        run_adversarial_experiment(
            "hedge", environment="historical_frequency_v3",
            feedback_mode="full_information", n_actions=actions, horizon=2,
            seed=7, output_dir=service.adversarial_raw_dir,
        )
    return app, service


def _fixed_url(service, player="0"):
    catalog = service.figure_builder.catalog("fixed")
    context = next(item for item in catalog["contexts"] if item["player"] == 0)
    return browse_url(service, player=player, feedback="both", horizon="2",
                      profiles=[profile["id"] for profile in context["profiles"]])


def _one_url(service):
    return browse_url(service, "adversarial", comparison_mode="actions",
                      action="all", horizon="2")


@pytest.mark.parametrize("mode,total", [("fixed", 53), ("adversarial", 27)])
def test_complete_group_pages_and_union(populated, mode, total):
    app, service = populated
    client = app.test_client()
    url = _fixed_url(service, "all") if mode == "fixed" else _one_url(service)
    catalog = service.figure_builder.catalog(mode)
    query, _, _ = parse_browsing_query(_params(url), catalog, mode)
    projection = project_dashboard_query(service.result_snapshot(mode), catalog, query)
    assert len(projection.groups) == total
    assert len(set(projection.group_ids)) == total

    expected_per_group = 2 if mode == "fixed" else 1
    collected = []
    for page_number in range(1, (total + 24) // 25 + 1):
        page_url = "/?" + query_string(query, page_number)
        response = client.get(page_url)
        assert response.status_code == 200
        ids = _group_ids(response)
        unique = list(dict.fromkeys(ids))
        assert len(unique) <= 25
        assert len(ids) == len(unique) * expected_per_group
        assert all(ids.count(group_id) == expected_per_group for group_id in unique)
        assert unique == list(projection.group_ids[(page_number - 1) * 25:page_number * 25])
        collected.extend(unique)
        assert f"Page {page_number} of {(total + 24) // 25}".encode() in response.data
        assert f"{total} scientific groups".encode() in response.data
        assert b'"runs":' not in response.data
        if page_number == 1:
            assert b'aria-label="Result pages"' in response.data
            assert b'name="page_size"' in response.data
            assert b'value="25" selected' in response.data
    assert collected == list(projection.group_ids)
    assert len(set(collected)) == total

    for size in (50, 100):
        response = client.get("/?" + query_string(query, 1, size))
        assert response.status_code == 200
        assert len(set(_group_ids(response))) == min(size, total)
        assert len(_rows(response)) == min(size, total) * expected_per_group
    last = client.get("/?" + query_string(query, 999), follow_redirects=False)
    assert last.status_code == 302
    assert _params(last.headers["Location"])["page"] == str((total + 24) // 25)
    assert client.get(last.headers["Location"]).status_code == 200
    assert client.get("/?" + query_string(query, 0)).status_code == 400
    assert client.get("/?" + query_string(query, 1, 101)).status_code == 400


def test_results_shrink_clamps_last_page(tmp_path, monkeypatch):
    app, service = create_test_app(tmp_path)
    for horizon in range(2, 29):
        run_cross_play_experiment(
            "rps", ("hedge", "hedge"), feedback_mode="full_information",
            horizon=horizon, seed=7, output_dir=service.raw_dir,
        )
    url = browse_url(service, comparison_mode="horizons", player="0")
    catalog = service.figure_builder.catalog("fixed")
    query, _, _ = parse_browsing_query(_params(url), catalog, "fixed")
    second_page = "/?" + query_string(query, 2)
    client = app.test_client()
    assert len(_rows(client.get(second_page))) == 2
    original = service.result_snapshot()
    removed = {group.records[0].group_id for group in original.groups("dashboard")[-3:]}
    shrunken = ResultSet(tuple(record for record in original.records
                              if record.group_id not in removed))
    monkeypatch.setattr(service, "result_snapshot",
                        lambda mode="fixed": shrunken if mode == "fixed" else original)
    response = client.get(second_page)
    assert response.status_code == 302
    assert _params(response.headers["Location"])["page"] == "1"
    assert len(_rows(client.get(response.headers["Location"]))) == 24


def test_sorting_empty_state_and_pager_state(populated):
    app, service = populated
    client = app.test_client()
    url = _fixed_url(service)
    catalog = service.figure_builder.catalog("fixed")
    query, _, _ = parse_browsing_query(_params(url), catalog, "fixed")
    sorted_query = replace(query, sort="average_external", direction="desc")
    projected = project_dashboard_query(
        service.result_snapshot(), catalog, sorted_query,
        presentations=service.game_presentations,
    )
    response = client.get("/?" + query_string(sorted_query, 2))
    assert response.status_code == 200
    assert _group_ids(response) == list(projected.group_ids[25:50])
    next_url = unescape(re.search(rb'<a href="([^"]+)">Next</a>', response.data).group(1).decode())
    assert _params(next_url)["sort"] == "average_external"
    assert b'data-sort="average_external"' in response.data
    display = client.get("/?" + query_string(replace(query, metric="external", view="average")))
    assert display.status_code == 200
    headers = re.findall(rb'<th\b[^>]*data-sort="(?:average|sqrt_scaling)_[^"]+"[^>]*>', display.data)
    rows = re.findall(rb'<tr\b(?=[^>]*class="summary-row")[^>]*>.*?</tr>', display.data, re.S)
    cells = [cell for row in rows for cell in re.findall(rb'<td\b[^>]*>', row)[6:12]]
    assert len(headers) == 6 and sum(b" hidden " in header for header in headers) == 5
    assert len(cells) == 25 * 6 and sum(b" hidden " in cell for cell in cells) == 25 * 5
    assert client.get("/?" + query_string(replace(query, player="all", sort="horizon"))).status_code == 400

    empty = replace(query, profiles=())
    response = client.get("/?" + query_string(empty))
    assert response.status_code == 200
    assert _rows(response) == []
    assert b"No results match this selection." in response.data
    assert b"Page 1 of 1" in response.data
    clamped = client.get("/?" + query_string(empty, 9))
    assert clamped.status_code == 302 and _params(clamped.headers["Location"])["page"] == "1"
    assert client.get("/?" + query_string(replace(query, context_id="stale"))).status_code == 400


def test_bootstrap_and_canonical_url_are_distinct(populated):
    app, service = populated
    client = app.test_client()
    for mode in ("fixed", "adversarial"):
        bare = client.get("/" if mode == "fixed" else "/?mode=adversarial")
        data = dashboard_data(bare)
        assert data["serverBrowsing"] is True and data["browsingBootstrap"] is True
        assert _rows(bare) == []
        assert data["browsingDefaultUrl"].startswith("/?mode=")
        canonical = client.get(_fixed_url(service) if mode == "fixed" else _one_url(service))
        data = dashboard_data(canonical)
        assert data["browsingBootstrap"] is False
        assert data["browsingState"]["context"]
        assert _rows(canonical)


def test_global_highlight_survives_slice_and_tied_best(populated, monkeypatch):
    app, service = populated
    catalog = service.figure_builder.catalog("fixed")
    url = _fixed_url(service)
    query, _, _ = parse_browsing_query(_params(url), catalog, "fixed")
    original = project_dashboard_query(service.result_snapshot(), catalog, query)
    tied_ids = {original.group_ids[0], original.group_ids[-1]}
    records = []
    for record in service.result_snapshot().records:
        values = [dict(player_values) for player_values in record.final_values]
        values[0]["average_external_regret"] = 0.1 if record.group_id in tied_ids else 1.0
        records.append(replace(record, final_values=tuple(values)))
    synthetic = ResultSet(tuple(records))
    display_query = replace(query, metric="external", view="average")
    projection = project_dashboard_query(synthetic, catalog, display_query)
    first = paginate_projection(projection, 1, 25)
    last = paginate_projection(projection, 3, 25)
    assert first.best_cells == last.best_cells == projection.best_cells
    assert {(group_id, player, column) for group_id, player, column in projection.best_cells
            if column == "average_external" and player == 0} == {
                (group_id, 0, "average_external") for group_id in tied_ids
            }
    assert original.group_ids[0] in {group.group_id for group in first.groups}
    assert original.group_ids[-1] in {group.group_id for group in last.groups}
    real_snapshot = service.result_snapshot
    monkeypatch.setattr(service, "result_snapshot",
                        lambda mode="fixed": synthetic if mode == "fixed" else real_snapshot(mode))
    for page_number, expected_id in ((1, original.group_ids[0]), (3, original.group_ids[-1])):
        response = app.test_client().get("/?" + query_string(display_query, page_number))
        assert response.status_code == 200
        row = re.search(
            rb'<tr\b[^>]*data-result-key="' + expected_id.encode() + rb'"[^>]*>.*?</tr>',
            response.data, re.S,
        )
        assert row is not None
        assert b'class="best-value"' in re.findall(rb'<td\b[^>]*>', row.group(0))[6]
    middle = app.test_client().get("/?" + query_string(display_query, 2))
    middle_rows = re.findall(rb'<tr\b(?=[^>]*class="summary-row")[^>]*>.*?</tr>', middle.data, re.S)
    assert all(b'class="best-value"' not in re.findall(rb'<td\b[^>]*>', row)[6]
               for row in middle_rows)


@pytest.mark.parametrize("mode,total", [("fixed", 53), ("adversarial", 27)])
def test_filtered_deletion_preview_is_page_and_sort_independent(populated, mode, total):
    app, service = populated
    client = app.test_client()
    base = _fixed_url(service, "0") if mode == "fixed" else _one_url(service)
    catalog = service.figure_builder.catalog(mode)
    query, _, _ = parse_browsing_query(_params(base), catalog, mode)
    token = csrf_token(client)
    previews = []
    for url in (
        "/?" + query_string(query, 1),
        "/?" + query_string(query, 2),
        "/?" + query_string(replace(query, sort="horizon", direction="desc"), 1, 50),
    ):
        page = client.get(url)
        assert page.status_code == 200
        state = dashboard_data(page)["browsingState"]
        body = {
            "_csrf_token": token, "mode": mode, "scope": state["scope"],
            "context": state["context"], "comparisonMode": state["comparisonMode"],
            "feedback": state["feedback"], "horizon": state["horizon"],
            "profiles": state["profiles"], "metric": state["metric"],
            "view": state["view"],
        }
        body["player" if mode == "fixed" else "action"] = (
            state["player"] if mode == "fixed" else state["action"])
        response = client.post(
            f"/experiment-groups/{mode}/delete-filtered/preview",
            data=body, headers={"Accept": "application/json"},
        )
        assert response.status_code == 200
        previews.append(response.get_json())
    assert previews[0] == previews[1] == previews[2]
    assert previews[0]["count"] == total


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_browser_filtered_deletion_uses_canonical_membership_state(populated, mode):
    app, service = populated
    base = _fixed_url(service, "0") if mode == "fixed" else _one_url(service)
    catalog = service.figure_builder.catalog(mode)
    query, _, _ = parse_browsing_query(_params(base), catalog, mode)
    page = app.test_client().get("/?" + query_string(query, 2)).get_data(as_text=True)
    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "page": page, "mode": mode,
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js")),
    }
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window;
w.fetch = async () => ({ok: true, json: async () => ({})});
w.eval(payload.script + "\nwindow.__state = filteredDeletionState();");
const state = w.__state;
assert.equal(state.mode, payload.mode);
assert(state.context);
assert(Array.isArray(state.profiles) && state.profiles.length);
assert(!("page" in state) && !("page_size" in state) && !("sort" in state));
assert(!("selections" in state) && !("resultKeys" in state));
assert.equal(payload.mode === "fixed" ? state.player : state.action,
             payload.mode === "fixed" ? "0" : "all");
dom.window.close();
'''
    run_node(script, payload=payload, jsdom=True)


def test_one_player_action_and_global_builder_catalog(populated):
    app, service = populated
    client = app.test_client()
    before = client.get("/figure-builder/options?mode=adversarial").data
    url = _one_url(service)
    page = client.get(url)
    assert len(_rows(page)) == 25
    row = re.search(rb'<tr\b(?=[^>]*class="summary-row")[^>]*>.*?</tr>', page.data, re.S)
    assert row is not None
    assert int(re.findall(rb'<td\b[^>]*>(.*?)</td>', row.group(0), re.S)[3]) >= 2
    assert b'historical_frequency_v3' in page.data
    assert client.get("/figure-builder/options?mode=adversarial").data == before
    assert len(service.figure_builder.catalog("adversarial")["contexts"][0]["result_keys"]) == 27


def test_explicit_empty_profile_url_remains_empty_in_builder(populated):
    app, service = populated
    catalog = service.figure_builder.catalog("fixed")
    query, _, _ = parse_browsing_query(_params(_fixed_url(service)), catalog, "fixed")
    empty_url = "/?" + query_string(replace(query, profiles=()))
    page = app.test_client().get(empty_url).get_data(as_text=True)
    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "page": page, "url": empty_url, "catalog": catalog,
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = r'''
const assert = require("assert").strict, {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const errors = new VirtualConsole();
errors.on("jsdomError", error => {throw error;});
const dom = new JSDOM(payload.page, {
    url: "http://localhost" + payload.url, runScripts: "outside-only", virtualConsole: errors,
});
const w = dom.window, nav = [];
w.__nav = nav;
w.localStorage.setItem("swap-regret-profile-batch-filters-fixed",
    JSON.stringify({scope: "rps", profiles: ["hedge_vs_hedge"]}));
w.fetch = async (url, options) => options
    ? {ok: true, json: async () => ({cached: false, figures: []})}
    : {ok: true, json: async () => payload.catalog};
const [dashboardCode, builder] = payload.script.split("\n/* One shared result selection");
w.eval(dashboardCode + "\nnavigateBrowsing = url => window.__nav.push(url);");
w.eval("/* One shared result selection" + builder);
(async () => {
    await new Promise(resolve => setImmediate(resolve));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(w.document.getElementById("filter-profiles").selectedOptions.length, 0);
    assert.equal(nav.length, 0);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_bare_bootstrap_saved_and_stale_state(tmp_path, mode):
    app, service = create_test_app(tmp_path)
    if mode == "fixed":
        for profile in (("hedge", "hedge"), ("ito_hedge", "ito_hedge")):
            run_cross_play_experiment(
                "rps", profile, feedback_mode="full_information", horizon=2,
                seed=4, output_dir=service.raw_dir,
            )
        scope, profiles, action = "rps", ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge"), ""
    else:
        for profile in ("hedge", "ito_hedge"):
            run_adversarial_experiment(
                profile, environment="historical_frequency_v3",
                feedback_mode="full_information", n_actions=2, horizon=2,
                seed=4, output_dir=service.adversarial_raw_dir,
            )
        scope, profiles, action = "historical_frequency_v3", ("hedge", "ito_hedge"), "2"
    catalog = service.figure_builder.catalog(mode)
    context = catalog["contexts"][0]
    key = f"{context['id']}:full_information:profiles:{action}:2"
    saved = {
        "scope": scope, "context": context["id"], "comparisonMode": "profiles",
        "feedback": "full_information", "horizon": "2", "player": "0",
        "selectedAction": action, "selectedHorizon": "2",
        "selections": {key: [profiles[1]]},
    }
    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "page": app.test_client().get("/" if mode == "fixed"
                                      else "/?mode=adversarial").get_data(as_text=True),
        "url": "/" if mode == "fixed" else "/?mode=adversarial",
        "mode": mode, "catalog": catalog, "saved": saved,
        "first": profiles[0], "second": profiles[1],
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = r'''
const assert = require("assert").strict, {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
    for (const [saved, expected] of [[null, payload.first], [payload.saved, payload.second],
                                     [{scope: "stale", context: "missing"}, payload.first]]) {
        const errors = new VirtualConsole();
        errors.on("jsdomError", error => {throw error;});
        const dom = new JSDOM(payload.page, {
            url: "http://localhost" + payload.url, runScripts: "outside-only",
            virtualConsole: errors,
        });
        const w = dom.window, nav = [];
        w.__nav = nav;
        if (saved) w.localStorage.setItem(
            "swap-regret-profile-batch-filters-" + payload.mode, JSON.stringify(saved));
        w.fetch = async (url, options) => options
            ? {ok: true, json: async () => ({cached: false, figures: []})}
            : {ok: true, json: async () => payload.catalog};
        const [dashboardCode, builder] = payload.script.split("\n/* One shared result selection");
        w.eval(dashboardCode + "\nnavigateBrowsing = url => window.__nav.push(url);");
        w.eval("/* One shared result selection" + builder);
        await tick(); await tick();
        assert.equal(nav.length, 1);
        const params = new URL(nav[0], w.location).searchParams;
        assert.equal(params.get("mode"), payload.mode);
        assert.deepEqual(params.getAll("profile"), [expected]);
        assert.equal(params.get("page"), "1");
        await tick();
        assert.equal(nav.length, 1);
        dom.window.close();
    }
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)


def test_browser_url_precedence_and_navigation(tmp_path):
    app, service = create_test_app(tmp_path)
    for profile in (("hedge", "hedge"), ("ito_hedge", "ito_hedge")):
        run_cross_play_experiment(
            "rps", profile, feedback_mode="full_information", horizon=2,
            seed=4, output_dir=service.raw_dir,
        )
    base = browse_url(service, player="0", profiles=["hedge_vs_hedge"])
    catalog = service.figure_builder.catalog("fixed")
    query, _, _ = parse_browsing_query(_params(base), catalog, "fixed")
    url = "/?" + query_string(query, 1, 50)
    page = app.test_client().get(url).get_data(as_text=True)
    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "page": page, "url": url, "catalog": service.figure_builder.catalog("fixed"),
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = r'''
const assert = require("assert").strict, {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const errors = new VirtualConsole();
errors.on("jsdomError", error => {throw error;});
const dom = new JSDOM(payload.page, {
    url: "http://localhost" + payload.url, runScripts: "outside-only", virtualConsole: errors,
});
const w = dom.window, d = w.document, navigations = [];
w.__nav = navigations;
w.localStorage.setItem("swap-regret-profile-batch-filters-fixed", JSON.stringify({
    scope: "rps", profiles: ["ito_hedge_vs_ito_hedge"], comparisonMode: "profiles",
    selections: {"other-context": ["ito_hedge_vs_ito_hedge"]},
}));
w.fetch = async (url, options) => options
    ? {ok: true, json: async () => ({cached: false, figures: []})}
    : {ok: true, json: async () => payload.catalog};
const [dashboardCode, builder] = payload.script.split("\n/* One shared result selection");
w.eval(dashboardCode + "\nwindow.__revision = () => filterSelectionRevision;" +
    "\nnavigateBrowsing = url => window.__nav.push(url);");
w.eval("/* One shared result selection" + builder);
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
    await tick(); await tick();
    const rows = () => [...d.querySelectorAll(".summary-row")].map(row => row.outerHTML);
    const initialRows = rows();
    assert(initialRows.length > 0);
    const profile = d.getElementById("filter-profiles");
    assert.equal(profile.value, "hedge_vs_hedge"); // URL beats localStorage.
    assert.deepEqual(JSON.parse(w.localStorage.getItem(
        "swap-regret-profile-batch-filters-fixed")).selections["other-context"],
        ["ito_hedge_vs_ito_hedge"]);
    assert.equal(navigations.length, 0);
    assert.equal(w.__revision(), 0); // Cache/status refresh is not a filter change.
    profile.value = "ito_hedge_vs_ito_hedge";
    profile.dispatchEvent(new w.Event("change"));
    await tick(); await tick(); // Async Builder status must not navigate again.
    assert.equal(navigations.length, 1);
    assert.equal(w.__revision(), 1);
    assert.deepEqual(rows(), initialRows); // Navigation, not DOM filtering or highlight recalculation.
    const params = new URL(navigations[0], w.location).searchParams;
    assert.equal(params.get("page"), "1");
    assert.equal(params.get("page_size"), "50");
    assert.deepEqual(params.getAll("profile"), ["ito_hedge_vs_ito_hedge"]);
    d.querySelector('th[data-sort="horizon"]').click();
    assert.equal(navigations.length, 2);
    const sorted = new URL(navigations[1], w.location).searchParams;
    assert.equal(sorted.get("sort"), "horizon");
    assert.equal(sorted.get("dir"), "asc");
    assert.equal(sorted.get("page"), "1");
    assert.deepEqual(rows(), initialRows); // Sorting does not reorder the current page.
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)


def _phase3e_case(tmp_path, mode, *, count=2, horizons=(2,)):
    app, service = create_test_app(tmp_path)
    algorithms = ("hedge", "ito_hedge", "bm_hedge")[:count]
    for algorithm in algorithms:
        for horizon in horizons:
            if mode == "fixed":
                run_cross_play_experiment(
                    "rps", (algorithm, algorithm), feedback_mode="full_information",
                    horizon=horizon, seed=4, output_dir=service.raw_dir,
                )
            else:
                run_adversarial_experiment(
                    algorithm, environment="historical_frequency_v3",
                    feedback_mode="full_information", n_actions=2, horizon=horizon,
                    seed=4, output_dir=service.adversarial_raw_dir,
                )
    profiles = [algorithm + "_vs_" + algorithm if mode == "fixed" else algorithm
                for algorithm in algorithms]
    return app, service, profiles


def _phase3e_preview(client, mode, state):
    body = {
        "_csrf_token": csrf_token(client), "mode": mode, "scope": state["scope"],
        "context": state["context"], "comparisonMode": state["comparisonMode"],
        "feedback": state["feedback"], "horizon": state["horizon"],
        "profiles": state["profiles"], "metric": state["metric"], "view": state["view"],
        "player" if mode == "fixed" else "action":
            state["player"] if mode == "fixed" else state["action"],
    }
    response = client.post(
        "/experiment-groups/" + mode + "/delete-filtered/preview",
        data=body, headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    return response.get_json()


_PHASE3E_BROWSER = r'''
const assert = require("assert").strict;
const {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const [dashboardCode, builderCode] = payload.script.split("\n/* One shared result selection");
const tick = () => new Promise(resolve => setImmediate(resolve));
async function render(page, url, saved = null) {
    const errors = new VirtualConsole();
    errors.on("jsdomError", error => {throw error;});
    const dom = new JSDOM(page, {
        url: "http://localhost" + url, runScripts: "outside-only", virtualConsole: errors,
    });
    const w = dom.window, d = w.document, nav = [], requests = [];
    w.__nav = nav;
    if (saved !== null) w.localStorage.setItem(
        "swap-regret-profile-batch-filters-" + payload.mode, saved);
    w.fetch = async (url, options) => {
        if (options) {
            requests.push({url: String(url), body: options.body});
            return {ok: true, json: async () => ({cached: false, figures: []})};
        }
        return {ok: true, json: async () => payload.catalog};
    };
    w.eval(dashboardCode + "\nwindow.__deletionState = filteredDeletionState;" +
        "\nnavigateBrowsing = url => window.__nav.push(url);");
    w.eval("/* One shared result selection" + builderCode);
    await tick(); await tick();
    return {dom, w, d, nav, requests,
        saved: w.localStorage.getItem("swap-regret-profile-batch-filters-" + payload.mode)};
}
'''.strip()


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
@pytest.mark.parametrize("comparison", ["regrets", "horizons"])
def test_all_profiles_canonical_url_refresh_and_history(tmp_path, mode, comparison):
    horizons = (2, 3) if comparison == "horizons" else (2,)
    app, service, profiles = _phase3e_case(tmp_path, mode, horizons=horizons)
    options = {
        "comparison_mode": comparison, "profiles": profiles,
        "player": "0", "action": "2", "feedback": "full_information",
        "horizon": "all" if comparison == "horizons" else "2",
    }
    all_url = browse_url(service, mode, **options)
    single_url = browse_url(service, mode, **(options | {"profiles": profiles[:1]}))
    client = app.test_client()
    all_page = client.get(all_url)
    single_page = client.get(single_url)
    assert all_page.status_code == single_page.status_code == 200
    assert len(set(_group_ids(all_page))) == len(profiles) * len(horizons)
    state = dashboard_data(all_page)["browsingState"]
    assert set(state["profiles"]) == set(profiles)
    assert _phase3e_preview(client, mode, state)["count"] == len(profiles) * len(horizons)

    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "mode": mode, "catalog": service.figure_builder.catalog(mode),
        "allPage": all_page.get_data(as_text=True), "singlePage": single_page.get_data(as_text=True),
        "allUrl": all_url, "singleUrl": single_url, "profiles": profiles,
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = _PHASE3E_BROWSER + r'''
(async () => {
    let single = await render(payload.singlePage, payload.singleUrl);
    assert.equal(single.d.getElementById("filter-profiles").value, payload.profiles[0]);
    assert.equal(single.nav.length, 0);
    single.d.getElementById("filter-profiles").value = "all";
    single.d.getElementById("filter-profiles").dispatchEvent(new single.w.Event("change"));
    await tick(); await tick();
    assert.equal(single.nav.length, 1);
    const next = new URL(single.nav[0], single.w.location).searchParams;
    assert.deepEqual(next.getAll("profile").sort(), [...payload.profiles].sort());
    assert.equal(next.get("page"), "1");
    const saved = single.w.localStorage.getItem(
        "swap-regret-profile-batch-filters-" + payload.mode);
    single.dom.window.close();

    let all = await render(payload.allPage, payload.allUrl, saved);
    assert.equal(all.nav.length, 0);
    assert.equal(all.d.getElementById("filter-profiles").value, "all");
    assert.deepEqual([...all.w.__deletionState().profiles].sort(),
        [...payload.profiles].sort());
    assert.deepEqual([...new all.w.FormData(all.d.getElementById("figure-builder")).getAll("profiles")],
        ["all"]);
    assert.equal(all.d.getElementById("builder-generate").disabled, false);
    const allSaved = all.saved;
    all.dom.window.close();

    let refresh = await render(payload.allPage, payload.allUrl, allSaved);
    assert.equal(refresh.d.getElementById("filter-profiles").value, "all");
    assert.equal(refresh.nav.length, 0);
    const refreshedSaved = refresh.saved;
    refresh.dom.window.close();

    let back = await render(payload.singlePage, payload.singleUrl, refreshedSaved);
    assert.equal(back.d.getElementById("filter-profiles").value, payload.profiles[0]);
    assert.equal(back.nav.length, 0);
    const backSaved = back.saved;
    back.dom.window.close();

    let forward = await render(payload.allPage, payload.allUrl, backSaved);
    assert.equal(forward.d.getElementById("filter-profiles").value, "all");
    assert.equal(forward.nav.length, 0);
    forward.dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_unsupported_multi_profile_single_select_url_is_rejected(tmp_path, mode):
    app, service, profiles = _phase3e_case(tmp_path, mode, count=3)
    options = {"player": "0", "action": "2", "horizon": "2",
               "feedback": "full_information"}
    client = app.test_client()
    subset = browse_url(service, mode, comparison_mode="regrets",
                        profiles=profiles[:2], **options)
    response = client.get(subset)
    assert response.status_code == 400
    assert b"multiple profiles are unsupported" in response.data
    assert client.get(browse_url(service, mode, comparison_mode="regrets",
                                 profiles=profiles, **options)).status_code == 200
    assert client.get(browse_url(service, mode, comparison_mode="regrets",
                                 profiles=profiles[:1], **options)).status_code == 200
    assert client.get(browse_url(service, mode, comparison_mode="profiles",
                                 profiles=profiles[:2], **options)).status_code == 200
    if mode == "adversarial":
        actions = browse_url(service, mode, comparison_mode="actions",
                             profiles=profiles[:2], action="all",
                             feedback="full_information", horizon="2")
        assert client.get(actions).status_code == 400


def test_fixed_all_player_url_refresh_history_builder_and_deletion(tmp_path):
    app, service, profiles = _phase3e_case(tmp_path, "fixed")
    options = {"comparison_mode": "profiles", "profiles": profiles,
               "feedback": "full_information", "horizon": "2"}
    all_url = browse_url(service, "fixed", player="all", **options)
    numbered_url = browse_url(service, "fixed", player="0", **options)
    client = app.test_client()
    all_page = client.get(all_url)
    numbered_page = client.get(numbered_url)
    assert all_page.status_code == numbered_page.status_code == 200
    assert len(_rows(all_page)) == 2 * len(_rows(numbered_page))
    all_state = dashboard_data(all_page)["browsingState"]
    numbered_state = dashboard_data(numbered_page)["browsingState"]
    assert all_state["player"] == "all"
    assert _phase3e_preview(client, "fixed", all_state)["count"] == len(set(_group_ids(all_page)))
    assert _phase3e_preview(client, "fixed", numbered_state)["count"] == len(
        set(_group_ids(numbered_page)))
    catalog = service.figure_builder.catalog("fixed")
    query, _, _ = parse_browsing_query(_params(all_url), catalog, "fixed")
    assert client.get("/?" + query_string(replace(query, sort="horizon"))).status_code == 400

    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "mode": "fixed", "catalog": catalog, "profiles": profiles,
        "allPage": all_page.get_data(as_text=True),
        "numberedPage": numbered_page.get_data(as_text=True),
        "allUrl": all_url, "numberedUrl": numbered_url,
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = _PHASE3E_BROWSER + r'''
(async () => {
    let all = await render(payload.allPage, payload.allUrl,
        JSON.stringify({player: "0", profiles: [payload.profiles[0]]}));
    const player = all.d.getElementById("filter-player");
    assert.equal(player.value, "all");
    assert.equal(player.selectedOptions[0].textContent, "All players");
    assert.equal(all.d.getElementById("builder-generate").disabled, true);
    assert.match(all.d.getElementById("builder-status").textContent, /numbered player/);
    assert.equal(all.requests.length, 0); // No player-specific Builder cache probe.
    assert.equal(all.w.__deletionState().player, "all");
    assert.deepEqual([...all.w.__deletionState().profiles].sort(), [...payload.profiles].sort());
    assert.equal(all.nav.length, 0);
    all.d.querySelector('th[data-sort="horizon"]').click();
    assert.equal(all.nav.length, 0); // All-player sorting stays disabled.
    all.d.getElementById("filter-view").value = "average";
    all.d.getElementById("filter-view").dispatchEvent(new all.w.Event("change"));
    assert.equal(all.nav.length, 1);
    const changed = new URL(all.nav[0], all.w.location).searchParams;
    assert.equal(changed.get("player"), "all");
    assert.equal(changed.get("view"), "average");
    assert.equal(changed.get("page"), "1");
    assert.deepEqual(changed.getAll("profile").sort(), [...payload.profiles].sort());
    const saved = all.w.localStorage.getItem("swap-regret-profile-batch-filters-fixed");
    all.dom.window.close();

    let refresh = await render(payload.allPage, payload.allUrl, saved);
    assert.equal(refresh.d.getElementById("filter-player").value, "all");
    assert.equal(refresh.d.getElementById("builder-generate").disabled, true);
    assert.equal(refresh.nav.length, 0);
    const refreshed = refresh.saved;
    refresh.dom.window.close();

    let numbered = await render(payload.numberedPage, payload.numberedUrl, refreshed);
    assert.equal(numbered.d.getElementById("filter-player").value, "0");
    assert.equal(numbered.nav.length, 0);
    const numberedSaved = numbered.saved;
    numbered.dom.window.close();

    let back = await render(payload.allPage, payload.allUrl, numberedSaved);
    assert.equal(back.d.getElementById("filter-player").value, "all");
    assert.equal(back.nav.length, 0);
    back.d.getElementById("filter-player").value = "0";
    back.d.getElementById("filter-player").dispatchEvent(new back.w.Event("change"));
    assert.equal(back.nav.length, 1);
    assert.equal(new URL(back.nav[0], back.w.location).searchParams.get("player"), "0");
    back.dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_hidden_regret_sort_clears_without_changing_membership(populated, mode):
    app, service = populated
    client = app.test_client()
    base = _fixed_url(service, "0") if mode == "fixed" else _one_url(service)
    catalog = service.figure_builder.catalog(mode)
    query, _, _ = parse_browsing_query(_params(base), catalog, mode)
    sorted_query = replace(query, metric="external", view="all",
                           sort="average_external", direction="desc")
    sorted_url = "/?" + query_string(sorted_query, 2)
    sorted_page = client.get(sorted_url)
    assert sorted_page.status_code == 200
    hidden_query = replace(sorted_query, metric="internal")
    hidden_response = client.get("/?" + query_string(hidden_query, 2), follow_redirects=False)
    assert hidden_response.status_code == 302
    cleared_url = hidden_response.headers["Location"]
    assert _params(cleared_url).get("sort") is None
    assert _params(cleared_url).get("dir") is None
    assert _params(cleared_url)["page"] == "2"
    cleared_page = client.get(cleared_url)
    assert cleared_page.status_code == 200
    cleared_state = dashboard_data(cleared_page)["browsingState"]
    assert cleared_state["metric"] == "internal"
    assert set(cleared_state["profiles"]) == set(query.profiles)
    unrelated_query = replace(sorted_query, sort="horizon")
    unrelated_url = "/?" + query_string(unrelated_query, 2)
    unrelated_page = client.get(unrelated_url)
    assert unrelated_page.status_code == 200

    snapshot = service.result_snapshot(mode)
    original = project_dashboard_query(snapshot, catalog, sorted_query)
    cleared = project_dashboard_query(snapshot, catalog,
                                      replace(hidden_query, sort=None, direction="asc"))
    assert set(original.group_ids) == set(cleared.group_ids)
    assert _phase3e_preview(client, mode, dashboard_data(sorted_page)["browsingState"]) == (
        _phase3e_preview(client, mode, cleared_state))

    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "mode": mode, "catalog": catalog,
        "sortedPage": sorted_page.get_data(as_text=True), "sortedUrl": sorted_url,
        "clearedPage": cleared_page.get_data(as_text=True), "clearedUrl": cleared_url,
        "unrelatedPage": unrelated_page.get_data(as_text=True), "unrelatedUrl": unrelated_url,
        "profiles": list(query.profiles),
        "script": "\n".join((static / name).read_text() for name in
                           ("common.js", "dashboard.js", "figure_builder.js")),
    }
    script = _PHASE3E_BROWSER + r'''
(async () => {
    let visible = await render(payload.sortedPage, payload.sortedUrl);
    assert.equal(visible.nav.length, 0);
    assert.equal(visible.d.querySelector('th[data-sort="average_external"]').dataset.direction,
        "descending");
    const beforeProfiles = [...new visible.w.FormData(
        visible.d.getElementById("figure-builder")).getAll("profiles")].sort();
    visible.d.getElementById("filter-view").value = "average";
    visible.d.getElementById("filter-view").dispatchEvent(new visible.w.Event("change"));
    assert.equal(visible.nav.length, 1);
    const compatible = new URL(visible.nav[0], visible.w.location).searchParams;
    assert.equal(compatible.get("sort"), "average_external");
    assert.equal(compatible.get("dir"), "desc");
    assert.equal(compatible.get("page"), "1");
    visible.dom.window.close();

    let metricHidden = await render(payload.sortedPage, payload.sortedUrl);
    metricHidden.d.getElementById("filter-metric").value = "internal";
    metricHidden.d.getElementById("filter-metric").dispatchEvent(
        new metricHidden.w.Event("change"));
    assert.equal(metricHidden.nav.length, 1);
    const reset = new URL(metricHidden.nav[0], metricHidden.w.location).searchParams;
    assert.equal(reset.get("sort"), null);
    assert.equal(reset.get("dir"), null);
    assert.equal(reset.get("page"), "1");
    assert.deepEqual(reset.getAll("profile").sort(), [...payload.profiles].sort());
    const saved = metricHidden.w.localStorage.getItem(
        "swap-regret-profile-batch-filters-" + payload.mode);
    metricHidden.dom.window.close();

    let cleared = await render(payload.clearedPage, payload.clearedUrl, saved);
    assert.equal(cleared.nav.length, 0);
    assert.equal(cleared.d.getElementById("filter-metric").value, "internal");
    assert.deepEqual([...new cleared.w.FormData(
        cleared.d.getElementById("figure-builder")).getAll("profiles")].sort(), beforeProfiles);
    assert.deepEqual([...cleared.w.__deletionState().profiles].sort(), [...payload.profiles].sort());
    cleared.dom.window.close();

    let viewHidden = await render(payload.sortedPage, payload.sortedUrl);
    viewHidden.d.getElementById("filter-view").value = "sqrt_scaling";
    viewHidden.d.getElementById("filter-view").dispatchEvent(new viewHidden.w.Event("change"));
    assert.equal(viewHidden.nav.length, 1);
    const viewReset = new URL(viewHidden.nav[0], viewHidden.w.location).searchParams;
    assert.equal(viewReset.get("sort"), null);
    assert.equal(viewReset.get("dir"), null);
    viewHidden.dom.window.close();

    let unrelated = await render(payload.unrelatedPage, payload.unrelatedUrl);
    unrelated.d.getElementById("filter-metric").value = "internal";
    unrelated.d.getElementById("filter-metric").dispatchEvent(new unrelated.w.Event("change"));
    assert.equal(unrelated.nav.length, 1);
    const preserved = new URL(unrelated.nav[0], unrelated.w.location).searchParams;
    assert.equal(preserved.get("sort"), "horizon");
    assert.equal(preserved.get("dir"), "desc");
    assert.equal(preserved.get("page"), "1");
    unrelated.dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_node(script, payload=payload, jsdom=True)
