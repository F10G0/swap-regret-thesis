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

from experiments.plots.style import profile_label
from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from tests.web.support import create_test_app, csrf_token, record_fixed_runs as result
from tests.support import read_csv_rows
from web.validation import FigureSelection, parse_figure_selection, parse_profile_selection


def form(context, profiles, **changes):
    return {"mode": context["mode"], "context_id": context["id"], "metric": "swap", "view": "average",
            "profiles": profiles} | changes


def context_for(service, mode="fixed", **criteria):
    return next(context for context in service.figure_builder.catalog(mode)["contexts"]
                if all(context[key] == value for key, value in criteria.items()))


def digest_files(directory):
    return {str(path): (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in directory.rglob("*.csv")}


def test_multi_select_is_canonical_and_never_defaults_to_all():
    data = MultiDict([("mode", "fixed"), ("context_id", "a" * 24), ("metric", "swap"), ("view", "average"),
                      ("profiles", "ito_vs_ito"), ("profiles", "hedge_vs_hedge"), ("profiles", "ito_vs_ito")])
    selection = parse_figure_selection(data)
    assert selection.profiles == ("hedge_vs_hedge", "ito_vs_ito")
    data.poplist("profiles")
    with pytest.raises(ValueError, match="at least one"):
        parse_figure_selection(data)


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"context_id": "../outside"}, {"metric": "unknown"},
    {"view": "unknown"}, {"profiles": ["../outside"]}, {"profiles": [None]},
])
def test_selection_rejects_invalid_inputs(changes):
    with pytest.raises(ValueError):
        parse_figure_selection(dict(mode="fixed", context_id="a" * 24, metric="swap", view="average", profiles=["hedge_vs_hedge"]) | changes)


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
    assert {p["label"] for p in compatible["profiles"]} == {"Hedge", "Ito", "Hedge vs Ito", "Ito vs Hedge"}
    assert {c["player"] for c in contexts} == {0, 1}
    assert all([p["id"] for p in c["profiles"]] == ["auer_exp3_vs_bm"] for c in contexts if c["feedback_mode"] == "bandit")
    assert str(service.raw_dir.resolve()) not in json.dumps(contexts)
    assert not service.figure_builder.output_dir.exists()


def test_cache_depends_on_subset_view_style_and_selected_file_state(tmp_path, monkeypatch):
    import experiments.plots.plot_regret as plotting
    import web.figure_builder as builder
    app, service = create_test_app(tmp_path)
    paths = result(service, "hedge_vs_hedge")
    result(service, "ito_vs_ito")
    context = context_for(service, player=0)
    build = lambda profiles, **kwargs: service.figure_builder.build(parse_figure_selection(form(context, profiles, **kwargs)))
    with pytest.raises(ValueError, match="at least one"):
        service.figure_builder.build(FigureSelection("fixed", context["id"], "swap", "average", ()))
    first = service.figure_builder.build(FigureSelection("fixed", context["id"], "swap", "average",
        ("ito_vs_ito", "hedge_vs_hedge", "ito_vs_ito")))
    assert first["profiles"] == ["hedge_vs_hedge", "ito_vs_ito"]
    output = service.figure_builder.output_dir / first["filename"]
    timestamp = output.stat().st_mtime_ns
    with monkeypatch.context() as patch:
        patch.setattr(plotting, "plot_regret", lambda *a, **k: pytest.fail("cache miss"))
        assert build(["ito_vs_ito", "hedge_vs_hedge"]) == first
    assert output.stat().st_mtime_ns == timestamp
    one = build(["hedge_vs_hedge"])
    assert one["artifact_id"] != first["artifact_id"]
    assert build(["hedge_vs_hedge"], view="sqrt_scaling")["artifact_id"] != one["artifact_id"]
    assert build(["hedge_vs_hedge"], metric="external")["artifact_id"] != one["artifact_id"]
    monkeypatch.setattr(builder, "PUBLICATION_STYLE_VERSION", builder.PUBLICATION_STYLE_VERSION + 1)
    styled = build(["hedge_vs_hedge"])
    assert styled["artifact_id"] != one["artifact_id"]
    paths[0].touch()
    assert build(["hedge_vs_hedge"])["artifact_id"] != styled["artifact_id"]


@pytest.mark.parametrize("mode,relative_paths", [("fixed", True), ("adversarial", False)])
def test_selected_png_pdf_routes_and_compatibility_boundaries(tmp_path, mode, relative_paths, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app, service = create_test_app(Path("results") if relative_paths else tmp_path)
    if mode == "fixed":
        profile = "auer_exp3_vs_bm"
        result(service, profile, mode="bandit")
        result(service, "hedge_vs_hedge")
    else:
        profile = "auer_exp3"
        for name in (profile, "ito"):
            for replicate in (0, 1):
                run_adversarial_experiment(name, n_actions=3, horizon=30, seed=42,
                    replicate=replicate, environment=RANDOM_WALK_ENVIRONMENT,
                    feedback_mode="bandit", output_dir=service.adversarial_raw_dir)
        run_adversarial_experiment("hedge", horizon=30, output_dir=service.adversarial_raw_dir)
    context = context_for(service, mode, player=0, feedback_mode="bandit")
    client = app.test_client()
    data = form(context, [profile]) | {"_csrf_token": csrf_token(client)}
    originals = digest_files(tmp_path)
    response = client.post("/figure-builder", data=data)
    assert response.status_code == 200
    artifact_path = service.figure_builder.artifact_path(response.json["filename"])
    assert artifact_path.is_absolute()
    image = client.get(response.json["url"])
    pdf = client.get(response.json["pdf_url"])
    assert image.status_code == pdf.status_code == 200
    assert image.mimetype == "image/png" and image.data.startswith(b"\x89PNG")
    assert pdf.mimetype == "application/pdf" and "attachment" in pdf.headers["Content-Disposition"]
    assert len(PdfReader(BytesIO(pdf.data)).pages) == 1
    assert client.post("/figure-builder", data=data | {"profiles": []}).status_code == 400
    assert client.post("/figure-builder", data=data | {"profiles": ["hedge_vs_hedge" if mode == "fixed" else "hedge"]}).status_code == 400
    assert client.post("/figure-builder", data=data | {"context_id": "b" * 24}).status_code == 400
    assert client.post("/figure-builder", data=form(context, [profile])).status_code == 400
    assert client.get("/figure-builder/files/../outside.pdf").status_code == 404
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private")
    (service.figure_builder.output_dir / "link.pdf").symlink_to(outside)
    assert client.get("/figure-builder/files/link.pdf").status_code == 404
    assert digest_files(tmp_path) == originals
    assert not service.jobs.recent()


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_collection_reads_once_exports_selected_means_and_reuses_cache(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    app, service = create_test_app(Path("results"))
    if mode == "fixed":
        import experiments.plots.plot_regret as plotting
        profiles = ["hedge_vs_ito", "ito_vs_hedge"]
        for profile in profiles + ["bm_vs_bm"]:
            result(service, profile)
        loader_name = "load_rows"
    else:
        import experiments.plots.plot_adversarial as plotting
        profiles = ["auer_exp3", "ito"]
        for name in profiles + ["bm"]:
            for replicate in (0, 1):
                run_adversarial_experiment(name, n_actions=3, horizon=30, seed=42,
                    feedback_mode="bandit", environment=RANDOM_WALK_ENVIRONMENT,
                    replicate=replicate, output_dir=service.adversarial_raw_dir)
        loader_name = "load_adversarial_rows"
    originals = digest_files(tmp_path)
    source_paths = {}
    loader = getattr(plotting, loader_name)
    reads, curves = Counter(), []
    original_save = plotting.save_figure_pair

    def load(path):
        reads[str(path)] += 1
        rows = loader(path)
        source_paths.setdefault(rows[0]["algorithm"], []).append(rows)
        return rows

    def save(figure, path, **kwargs):
        curves.append([line for line in figure.axes[0].lines if not line.get_label().startswith("_")])
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, loader_name, load)
    monkeypatch.setattr(plotting, "save_figure_pair", save)
    monkeypatch.setattr(service, "submit_experiment", lambda *a, **k: pytest.fail("reran experiment"))
    context = context_for(service, mode, player=0)
    client = app.test_client()
    data = {"mode": mode, "context_id": context["id"], "profiles": list(reversed(profiles)),
            "_csrf_token": csrf_token(client)}
    response = client.post("/figure-builder/collection", data=data)
    assert response.status_code == 200
    figures = response.json["figures"]
    assert [(figure["metric"], figure["view"]) for figure in figures] == [
        (metric, view) for metric in ("external", "internal", "swap") for view in ("average", "sqrt_scaling")]
    assert len(reads) == 4 and set(reads.values()) == {1}
    for figure, lines in zip(figures, curves):
        assert [line.get_label() for line in lines] == [profile_label(p.split("_vs_")) for p in profiles]
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
        assert figure["profiles"] == profiles
        preview, pdf = client.get(figure["url"]), client.get(figure["pdf_url"])
        assert preview.status_code == pdf.status_code == 200
        assert preview.data.startswith(b"\x89PNG")
        assert len(PdfReader(BytesIO(pdf.data)).pages) == 1
    timestamps = {p: p.stat().st_mtime_ns for p in service.figure_builder.output_dir.iterdir()}
    cached = client.post("/figure-builder/collection", data=data | {"profiles": profiles})
    assert cached.json == response.json
    assert set(reads.values()) == {1} and len(curves) == 6
    assert {p: p.stat().st_mtime_ns for p in timestamps} == timestamps
    chosen = [figures[5], figures[1], figures[3]]
    download = client.post("/figures/download-filtered.pdf", data={
        "mode": "figure_builder", "_csrf_token": data["_csrf_token"],
        "filenames": [figure["pdf_filename"] for figure in chosen],
    })
    assert download.status_code == 200 and download.mimetype == "application/pdf"
    pages = PdfReader(BytesIO(download.data)).pages
    assert len(pages) == 3
    for page, figure in zip(pages, chosen):
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


def test_profile_selection_does_not_need_regret_or_view():
    selection = parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
                                         "profiles": ["ito_vs_ito", "hedge_vs_hedge", "ito_vs_ito"]})
    assert selection.profiles == ("hedge_vs_hedge", "ito_vs_ito")


def test_context_identity_and_ordering_preserve_duplicate_policies(tmp_path):
    _, service = create_test_app(tmp_path)
    paths = result(service, "ito_vs_hedge", replicates=(2, 0))
    result(service, "hedge_vs_ito", replicates=(0, 2))
    row = read_csv_rows(paths[0])[0]
    key = ("rps", row["game_payoff_digest"], "full_information", 30, 42, row["stationary_method"],
           int(row["implementation_version"]), row["runtime_fingerprint"])
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
        output = args[-1] / "figure.png"
        output.write_bytes(b"png")
        output.with_suffix(".pdf").write_bytes(b"pdf")
        stat = paths[0].stat()
        os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(plotting, "plot_regret", render)
    with pytest.raises(ValueError, match="Results changed during rendering"):
        service.figure_builder.build(parse_figure_selection(form(context, ["hedge_vs_hedge"])))
    assert not list(service.figure_builder.output_dir.iterdir())
