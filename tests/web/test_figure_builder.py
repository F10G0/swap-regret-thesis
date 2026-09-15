from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import numpy as np
from pypdf import PdfReader
import pytest
from werkzeug.datastructures import MultiDict

from experiments.algorithm_labels import algorithm_profile_label
from experiments.plots.style import regret_series_style
from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from tests.web.support import create_test_app, csrf_token, record_fixed_runs as result
from tests.support import read_csv_rows
from web.validation import parse_profile_selection


def form(context, profiles, comparison_mode="profiles", metric="external", view="all", action=None):
    values = {"mode": context["mode"], "context_id": context["id"], "comparison_mode": comparison_mode,
              "metric": metric, "view": view, "profiles": profiles}
    if context["mode"] == "adversarial":
        values["action"] = str(action or context["actions"][0])
    return values


def context_for(service, mode="fixed", **criteria):
    return next(context for context in service.figure_builder.catalog(mode)["contexts"]
                if all(context[key] == value for key, value in criteria.items()))


def digest_files(directory):
    return {str(path): (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in directory.rglob("*.csv")}


def test_comparison_selection_is_canonical_and_enforces_mode_contracts():
    data = MultiDict([("mode", "fixed"), ("context_id", "a" * 24), ("metric", "external"), ("profiles", "ito_vs_ito"),
                      ("profiles", "hedge_vs_hedge"), ("profiles", "ito_vs_ito")])
    selection = parse_profile_selection(data)
    assert selection.profiles == ("hedge_vs_hedge", "ito_vs_ito")
    assert (selection.comparison_mode, selection.metric, selection.view) == ("profiles", "external", "all")
    with pytest.raises(ValueError, match="exactly one"):
        parse_profile_selection(data | {"comparison_mode": "regrets"})
    regret_selection = parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
                                                 "comparison_mode": "regrets", "profiles": ["hedge_vs_hedge"]})
    assert (regret_selection.metric, regret_selection.view) == ("all", "all")
    with pytest.raises(ValueError, match="at least one"):
        parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
                                 "comparison_mode": "regrets", "profiles": []})
    with pytest.raises(ValueError, match="all regret notions"):
        parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
                                 "comparison_mode": "regrets", "metric": "external",
                                 "profiles": ["hedge_vs_hedge"]})
    action_selection = parse_profile_selection({"mode": "adversarial", "context_id": "a" * 24,
        "comparison_mode": "actions", "metric": "swap", "profiles": ["hedge"], "action": "2"})
    assert action_selection.action == "all"
    with pytest.raises(ValueError, match="only for one-player"):
        parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
            "comparison_mode": "actions", "metric": "swap", "profiles": ["hedge_vs_hedge"]})
    data.poplist("profiles")
    with pytest.raises(ValueError, match="at least one"):
        parse_profile_selection(data)


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"context_id": "../outside"},
    {"profiles": ["../outside"]}, {"profiles": [None]},
    {"comparison_mode": "games"}, {"metric": "unknown"}, {"view": "unknown"},
])
def test_selection_rejects_invalid_inputs(changes):
    with pytest.raises(ValueError):
        parse_profile_selection(dict(mode="fixed", context_id="a" * 24, metric="external",
                                     profiles=["hedge_vs_hedge"]) | changes)


def test_options_use_real_groups_and_separate_incompatible_metadata(tmp_path):
    app, service = create_test_app(tmp_path)
    result(service, "hedge_vs_hedge")
    result(service, "ito_vs_ito")
    result(service, "hedge_vs_ito")
    result(service, "ito_vs_hedge")
    result(service, "bm_vs_bm", horizon=31)
    result(service, "regret_matching_vs_regret_matching", seed=7)
    result(service, "stationary_regret_matching_vs_stationary_regret_matching", replicates=(0,))
    result(service, "auer_exp3_vs_bm", mode="bandit")
    result(service, "hedge_vs_hedge", game="rpsls")
    response = app.test_client().get("/figure-builder/options?mode=fixed")
    assert response.status_code == 200
    contexts = response.json["contexts"]
    compatible = next(c for c in contexts if {p["id"] for p in c["profiles"]} == {
        "hedge_vs_hedge", "ito_vs_ito", "hedge_vs_ito", "ito_vs_hedge"})
    assert {p["label"] for p in compatible["profiles"]} == {
        "Hedge vs Hedge", "Ito vs Ito", "Hedge vs Ito", "Ito vs Hedge"}
    assert compatible["horizon"] == 30 and compatible["base_seed"] == 42
    assert {c["player"] for c in contexts} == {0, 1}
    assert all([p["id"] for p in c["profiles"]] == ["auer_exp3_vs_bm"] for c in contexts if c["feedback_mode"] == "bandit")
    assert str(service.raw_dir.resolve()) not in json.dumps(contexts)
    assert not service.figure_builder.output_dir.exists()


def test_cache_depends_on_subset_view_style_and_selected_file_state(tmp_path, monkeypatch):
    import experiments.plots.plot_regret as plotting
    import web.figure_builder as builder
    _, service = create_test_app(tmp_path)
    paths = result(service, "hedge_vs_hedge")
    result(service, "ito_vs_ito")
    context = context_for(service, player=0)

    def selection(profiles, metric="swap", view="average", comparison_mode="profiles"):
        return parse_profile_selection(form(context, profiles, comparison_mode, metric, view))

    def build(profiles, metric="swap", view="average", comparison_mode="profiles"):
        collection = service.figure_builder.build_collection(selection(profiles, metric, view, comparison_mode))
        return collection["figures"][0]

    with pytest.raises(ValueError, match="at least one"):
        build([])
    with monkeypatch.context() as patch:
        patch.setattr(plotting, "plot_regret_curves", lambda *a, **k: pytest.fail("cache probe rendered"))
        missing = service.figure_builder.cached_collection(selection(["ito_vs_ito", "hedge_vs_hedge"]))
    assert missing["cached"] is False and missing["figures"] == []
    assert not service.figure_builder.output_dir.exists()
    first = build(["ito_vs_ito", "hedge_vs_hedge", "ito_vs_ito"])
    assert first["profiles"] == ["hedge_vs_hedge", "ito_vs_ito"]
    output = service.figure_builder.output_dir / first["filename"]
    timestamp = output.stat().st_mtime_ns
    with monkeypatch.context() as patch:
        patch.setattr(plotting, "plot_regret_curves", lambda *a, **k: pytest.fail("cache miss"))
        cached = service.figure_builder.cached_collection(selection(["ito_vs_ito", "hedge_vs_hedge"]))
        assert cached["cached"] is True and cached["figures"] == [first]
    assert output.stat().st_mtime_ns == timestamp
    assert service.figure_builder.cached_collection(selection(["hedge_vs_hedge"]))["cached"] is False
    one = build(["hedge_vs_hedge"])
    assert one["artifact_id"] != first["artifact_id"]
    assert build(["hedge_vs_hedge"], view="sqrt_scaling")["artifact_id"] != one["artifact_id"]
    assert build(["hedge_vs_hedge"], metric="external")["artifact_id"] != one["artifact_id"]
    compared = build(["hedge_vs_hedge"], metric="all", comparison_mode="regrets")
    assert compared["artifact_id"] != one["artifact_id"]
    monkeypatch.setattr(builder, "PUBLICATION_STYLE_VERSION", builder.PUBLICATION_STYLE_VERSION + 1)
    styled = build(["hedge_vs_hedge"])
    assert styled["artifact_id"] != one["artifact_id"]
    paths[0].touch()
    assert build(["hedge_vs_hedge"])["artifact_id"] != styled["artifact_id"]


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_collection_reads_once_exports_selected_means_and_reuses_cache(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    app, service = create_test_app(Path("results"))
    import experiments.plots.plot_regret as fixed_plotting
    if mode == "fixed":
        plotting = fixed_plotting
        loader_module = plotting
        profiles = ["hedge_vs_hedge", "hedge_vs_ito"]
        for profile in profiles + ["bm_vs_bm"]:
            result(service, profile)
        loader_name = "load_rows"
    else:
        import experiments.plots.plot_adversarial as plotting
        from experiments.scenarios import adversarial as loader_module
        profiles = ["auer_exp3", "ito"]
        for name in profiles + ["bm"]:
            for replicate in (0, 1):
                run_adversarial_experiment(name, n_actions=3, horizon=30, seed=42,
                    feedback_mode="bandit", environment=RANDOM_WALK_ENVIRONMENT,
                    replicate=replicate, output_dir=service.adversarial_raw_dir)
        loader_name = "load_adversarial_rows"
    originals = digest_files(tmp_path)
    source_paths = {}
    loader = getattr(loader_module, loader_name)
    reads, curves = Counter(), []
    original_save = fixed_plotting.save_figure_pair

    def load(path):
        reads[str(path)] += 1
        rows = loader(path)
        source_paths.setdefault(rows[0]["algorithm"], []).append(rows)
        return rows

    def save(figure, path, **kwargs):
        curves.append([line for line in figure.axes[0].lines if not line.get_label().startswith("_")])
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(loader_module, loader_name, load)
    monkeypatch.setattr(fixed_plotting, "save_figure_pair", save)
    monkeypatch.setattr(service, "submit_experiment", lambda *a, **k: pytest.fail("reran experiment"))
    context = context_for(service, mode, player=0)
    client = app.test_client()
    data = form(context, list(reversed(profiles))) | {"_csrf_token": csrf_token(client)}
    response = client.post("/figure-builder/collection", data=data)
    assert response.status_code == 200
    figures = response.json["figures"]
    assert [(figure["metric"], figure["view"]) for figure in figures] == [
        ("external", view) for view in ("average", "sqrt_scaling")]
    assert len(reads) == 4 and set(reads.values()) == {1}
    styles = {}
    for figure, lines in zip(figures, curves):
        assert [line.get_label() for line in lines] == [algorithm_profile_label(p.split("_vs_")) for p in profiles]
        column = ("average_" if figure["view"] == "average" else "") + figure["metric"] + "_regret"
        scaled = figure["view"] == "sqrt_scaling"
        for profile, line in zip(profiles, lines):
            trajectories = source_paths[profile]
            assert len(trajectories) == 2
            if mode == "fixed":
                times, means = plotting.aggregate_metric_curve(trajectories, 0, column, divide_by_sqrt_time=scaled)
            else:
                times, means = plotting.aggregate_adversarial_regret(trajectories, column, scale_by_sqrt_time=scaled)
            np.testing.assert_array_equal(line.get_xdata(), times)
            np.testing.assert_array_equal(line.get_ydata(), means)
            style = (line.get_color(), line.get_linestyle(), line.get_marker(), line.get_markevery())
            styles.setdefault(profile, style)
            assert styles[profile] == style
        assert figure["profiles"] == profiles
        assert figure["comparison_mode"] == "profiles"
        preview, pdf = client.get(figure["url"]), client.get(figure["pdf_url"])
        assert preview.status_code == pdf.status_code == 200
        assert preview.data.startswith(b"\x89PNG")
        assert len(PdfReader(BytesIO(pdf.data)).pages) == 1
    timestamps = {p: p.stat().st_mtime_ns for p in service.figure_builder.output_dir.iterdir()}
    cached = client.post("/figure-builder/cache", data=data | {"profiles": profiles})
    assert cached.json == response.json
    assert set(reads.values()) == {1} and len(curves) == 2
    assert {style[1] for style in styles.values()} == {"-"}
    assert len({(style[0], style[2]) for style in styles.values()}) == len(profiles)
    assert [styles[profile][2] for profile in profiles] == ["o", "s"]
    assert [styles[profile][3] for profile in profiles] == [(0, 0.24), (0.12, 0.24)]
    assert {p: p.stat().st_mtime_ns for p in timestamps} == timestamps
    download = client.post("/figures/download-filtered.pdf", data={
        "mode": "figure_builder", "_csrf_token": data["_csrf_token"],
        "filenames": [figure["pdf_filename"] for figure in figures],
    })
    assert download.status_code == 200 and download.mimetype == "application/pdf"
    pages = PdfReader(BytesIO(download.data)).pages
    assert len(pages) == 2
    for page, figure in zip(pages, figures):
        original = PdfReader(service.figure_builder.artifact_path(figure["pdf_filename"])).pages[0]
        assert page.extract_text() == original.extract_text()
        assert not list(page.images)  # Merging keeps the publication PDFs vector-based.
    assert client.post("/figure-builder/collection", data=data | {"profiles": []}).status_code == 400
    assert client.post("/figure-builder/collection", data=data | {"profiles": ["unknown"]}).status_code == 400
    assert client.post("/figure-builder/collection", data=data | {"context_id": "a" * 24}).status_code == 400
    assert client.post("/figure-builder/collection", data={"mode": mode}).status_code == 400
    assert client.post("/figures/download-filtered.pdf", data={
        "mode": "figure_builder", "_csrf_token": data["_csrf_token"], "filenames": ["../private.pdf"],
    }).status_code == 404
    assert digest_files(tmp_path) == originals and not service.jobs.recent()


def test_action_space_comparison_uses_compatible_ordinary_trajectories(tmp_path, monkeypatch):
    import experiments.plots.plot_adversarial as plotting
    import experiments.plots.plot_regret as renderer
    from experiments.scenarios.adversarial import load_adversarial_rows

    app, service = create_test_app(tmp_path)
    paths = {}
    for n_actions in (2, 4):
        paths[n_actions] = [run_adversarial_experiment("hedge", n_actions=n_actions, horizon=30, seed=42,
            replicate=replicate, output_dir=service.adversarial_raw_dir) for replicate in (0, 1)]
    run_adversarial_experiment("hedge", n_actions=6, horizon=30, seed=9, output_dir=service.adversarial_raw_dir)
    context = context_for(service, "adversarial", base_seed=42)
    assert context["actions"] == [2, 4]
    assert context["profiles"] == [{"id": "hedge", "label": "Hedge",
                                     "metrics": ["external", "internal", "swap"], "actions": [2, 4]}]

    captured = []
    original_save = renderer.save_figure_pair
    def save(figure, path, **kwargs):
        captured.extend(line for line in figure.axes[0].lines if not line.get_label().startswith("_"))
        return original_save(figure, path, **kwargs)
    monkeypatch.setattr(renderer, "save_figure_pair", save)
    client = app.test_client()
    response = client.post("/figure-builder/collection", data=form(
        context, ["hedge"], "actions", "swap", "average", "all") | {"_csrf_token": csrf_token(client)})
    assert response.status_code == 200
    assert [line.get_label() for line in captured] == ["K=2", "K=4"]
    for line, n_actions in zip(captured, (2, 4)):
        expected = plotting.aggregate_adversarial_regret(
            [load_adversarial_rows(path) for path in paths[n_actions]], "average_swap_regret")
        np.testing.assert_array_equal(line.get_xdata(), expected[0])
        np.testing.assert_array_equal(line.get_ydata(), expected[1])
    figure = response.json["figures"][0]
    assert figure["comparison_mode"] == "actions" and "Action spaces" in figure["title"]


def test_regret_comparison_combines_three_fixed_series_for_selected_views_and_export(tmp_path, monkeypatch):
    import experiments.plots.plot_regret as plotting

    app, service = create_test_app(tmp_path)
    paths = result(service, "hedge_vs_hedge")
    trajectories = [plotting.load_rows(path) for path in paths]
    context = context_for(service, player=0)
    captured = []
    original_save = plotting.save_figure_pair

    def save(figure, path, **kwargs):
        captured.append((figure.axes[0].get_ylabel(), [line for line in figure.axes[0].lines
                                                       if not line.get_label().startswith("_")]))
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, "save_figure_pair", save)
    client = app.test_client()
    token = csrf_token(client)

    def build(view):
        response = client.post("/figure-builder/collection", data=form(
            context, ["hedge_vs_hedge"], "regrets", "all", view) | {"_csrf_token": token})
        assert response.status_code == 200
        return response

    average = build("average")
    scaling = build("sqrt_scaling")
    both = build("all")
    assert [len(response.json["figures"]) for response in (average, scaling, both)] == [1, 1, 2]
    assert [figure["view"] for figure in both.json["figures"]] == ["average", "sqrt_scaling"]
    assert len(captured) == 2
    styles = {}
    for (y_label, lines), view in zip(captured, ("average", "sqrt_scaling")):
        assert y_label == ("$R_T/T$" if view == "average" else "$R_T/\\sqrt{T}$")
        assert [line.get_label() for line in lines] == ["External regret", "Internal regret", "Swap regret"]
        for index, (metric, line) in enumerate(zip(("external", "internal", "swap"), lines)):
            column = ("average_" if view == "average" else "") + metric + "_regret"
            x, y = plotting.aggregate_metric_curve(trajectories, 0, column,
                                                   divide_by_sqrt_time=view == "sqrt_scaling")
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            style = (line.get_color(), line.get_linestyle(), line.get_marker(), line.get_markevery(), line.get_zorder())
            assert style == tuple(regret_series_style(metric, index, 3).values())
            styles.setdefault(metric, style)
            assert styles[metric] == style
    assert len(set(styles.values())) == 3
    assert [styles[metric][1] for metric in ("external", "swap", "internal")] == ["-", "--", "-."]
    assert [styles[metric][0] for metric in ("external", "swap", "internal")] == ["#0072B2", "#D55E00", "#CC79A7"]
    assert [styles[metric][3][0] for metric in ("external", "internal", "swap")] == pytest.approx([0, 0.08, 0.16])
    assert {styles[metric][3][1] for metric in ("external", "internal", "swap")} == {0.24}
    assert styles["swap"][4] > styles["external"][4]
    assert build("all").json == both.json
    assert len(captured) == 2

    for figures, expected_pages in ((average.json["figures"], 1), (both.json["figures"], 2)):
        response = client.post("/figures/download-filtered.pdf", data={
            "mode": "figure_builder", "_csrf_token": token,
            "filenames": [figure["pdf_filename"] for figure in figures],
        })
        assert response.status_code == 200
        assert len(PdfReader(BytesIO(response.data)).pages) == expected_pages


def test_context_identity_and_ordering_preserve_duplicate_policies(tmp_path):
    _, service = create_test_app(tmp_path)
    paths = result(service, "ito_vs_hedge", replicates=(2, 0))
    result(service, "hedge_vs_ito", replicates=(0, 2))
    row = read_csv_rows(paths[0])[0]
    key = ("rps", row["game_payoff_digest"], "full_information", 30, 42, row["stationary_method"],
           row["runtime_fingerprint"])
    contexts = {context["id"]: context for context in service.figure_builder.catalog("fixed")["contexts"]}
    for player in (0, 1):
        expected_id = sha256(json.dumps(("fixed", key, player, [0, 2]), sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()[:24]
        context = contexts[expected_id]
        assert [profile["id"] for profile in context["profiles"]] == ["hedge_vs_ito", "ito_vs_hedge"]
    groups = service.result_snapshot().groups("builder")
    assert next(group.paths for group in groups if group.records[0].profile == ("ito", "hedge")) == list(reversed(paths))
    for replicate in (2, 0):
        path = run_adversarial_experiment("hedge", n_actions=3, horizon=30, seed=42,
            replicate=replicate, output_dir=service.adversarial_raw_dir)
    assert len(service.figure_builder.catalog("adversarial")["contexts"]) == 1
    shutil.copyfile(path, service.adversarial_raw_dir / "duplicate.csv")
    assert len(service.result_snapshot("adversarial").records) == 3
    assert service.figure_builder.catalog("adversarial")["contexts"] == []


def test_builder_rejects_source_mutation_before_publication(tmp_path, monkeypatch):
    import experiments.plots.plot_regret as plotting
    _, service = create_test_app(tmp_path)
    paths = result(service, "hedge_vs_hedge")
    context = context_for(service, player=0)

    def render(*args):
        output = args[-1]
        output.write_bytes(b"png")
        output.with_suffix(".pdf").write_bytes(b"pdf")
        stat = paths[0].stat()
        os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(plotting, "plot_regret_curves", render)
    with pytest.raises(ValueError, match="Results changed during rendering"):
        service.figure_builder.build_collection(parse_profile_selection(form(context, ["hedge_vs_hedge"])))
    assert not list(service.figure_builder.output_dir.iterdir())
