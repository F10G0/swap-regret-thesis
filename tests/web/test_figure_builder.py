from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from pypdf import PdfReader
import pytest
from werkzeug.datastructures import MultiDict

from experiments.plots.style import algorithm_style, profile_label
from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from experiments.scenarios.bandit_cross_play import run_bandit_cross_play_experiment
from tests.web.support import create_test_app, csrf_token
from web.validation import FigureSelection, parse_figure_selection


def result(service, profile, *, mode="full_information", game="rps", horizon=30, seed=42, replicates=(0, 1)):
    runner = run_full_information_cross_play_experiment if mode == "full_information" else run_bandit_cross_play_experiment
    return [runner(game, profile.split("_vs_"), horizon=horizon, seed=seed, replicate=replicate,
                   output_dir=service.raw_dir, max_recorded_points=10) for replicate in replicates]


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
    assert not any("_paths" in context for context in contexts)
    assert not service.figure_builder.output_dir.exists()


@pytest.mark.parametrize("selected", [
    ["hedge_vs_hedge"], ["hedge_vs_hedge", "ito_vs_ito", "bm_vs_bm"],
    ["hedge_vs_ito", "ito_vs_hedge"],
])
def test_selected_curves_only_and_complete_replicate_means(tmp_path, monkeypatch, selected):
    import experiments.plots.plot_regret as plotting
    app, service = create_test_app(tmp_path)
    all_profiles = ["hedge_vs_hedge", "ito_vs_ito", "bm_vs_bm", "hedge_vs_ito", "ito_vs_hedge"]
    for profile in all_profiles:
        result(service, profile)
    originals = digest_files(tmp_path)
    context = context_for(service, player=0)
    captured = []
    original_save = plotting.save_figure_pair

    def save(figure, path, **kwargs):
        captured.extend(line for line in figure.axes[0].lines if not line.get_label().startswith("_"))
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, "save_figure_pair", save)
    monkeypatch.setattr(service, "submit_experiment", lambda *a, **k: pytest.fail("selection reran experiments"))
    import experiments.runner as runner
    monkeypatch.setattr(runner, "run_game", lambda *a, **k: pytest.fail("selection ran learner updates"))
    rendered = service.figure_builder.build(parse_figure_selection(form(context, selected)))
    assert [line.get_label() for line in captured] == [profile_label(p.split("_vs_")) for p in sorted(selected)]
    assert len(captured) == len(selected)
    for profile, line in zip(sorted(selected), captured):
        paths = service.figure_builder._contexts("fixed")[context["id"]]["_paths"][profile]
        times, means = plotting.aggregate_metric_curve([plotting.load_rows(p) for p in paths], 0, "average_swap_regret")
        np.testing.assert_array_equal(line.get_xdata(), times)
        np.testing.assert_array_equal(line.get_ydata(), means)
        assert line.get_color() == algorithm_style(profile.split("_vs_")[0])["color"]
    assert digest_files(tmp_path) == originals
    assert (service.figure_builder.output_dir / rendered["pdf_filename"]).is_file()
    assert not list(tmp_path.rglob(".plot-cache"))
    assert not service.jobs.recent()


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


def test_adversarial_selected_curves_keep_existing_means_and_style(tmp_path, monkeypatch):
    import experiments.plots.plot_adversarial as plotting
    app, service = create_test_app(tmp_path)
    paths = {}
    for name in ("auer_exp3", "bm", "ito"):
        paths[name] = [run_adversarial_experiment(name, n_actions=3, horizon=30, seed=42,
            replicate=replicate, environment=RANDOM_WALK_ENVIRONMENT,
            feedback_mode="bandit", output_dir=service.adversarial_raw_dir) for replicate in (0, 1)]
    originals = digest_files(tmp_path)
    captured = []
    original_save = plotting.save_figure_pair

    def save(figure, path, **kwargs):
        captured.extend(line for line in figure.axes[0].lines if not line.get_label().startswith("_"))
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, "save_figure_pair", save)
    import experiments.scenarios.adversarial as runner
    monkeypatch.setattr(runner, "run_adversarial_experiment", lambda *a, **k: pytest.fail("selection reran experiments"))
    context = context_for(service, "adversarial", feedback_mode="bandit")
    selected = ["auer_exp3", "ito"]
    service.figure_builder.build(parse_figure_selection(form(context, selected, view="sqrt_scaling")))
    assert [line.get_label() for line in captured] == [profile_label([name]) for name in selected]
    for name, line in zip(selected, captured):
        times, means = plotting.aggregate_adversarial_regret(
            [plotting.load_adversarial_rows(path) for path in paths[name]], "swap_regret", scale_by_sqrt_time=True)
        np.testing.assert_array_equal(line.get_xdata(), times)
        np.testing.assert_array_equal(line.get_ydata(), means)
        assert line.get_color() == algorithm_style(name)["color"]
        assert line.get_linestyle() == algorithm_style(name)["linestyle"]
    assert digest_files(tmp_path) == originals
    assert not service.jobs.recent()


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
@pytest.mark.parametrize("relative_paths", [False, True])
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
    timestamp = artifact_path.stat().st_mtime_ns
    image = client.get(response.json["url"])
    pdf = client.get(response.json["pdf_url"])
    assert image.status_code == pdf.status_code == 200
    assert image.mimetype == "image/png" and image.data.startswith(b"\x89PNG")
    assert pdf.mimetype == "application/pdf" and "attachment" in pdf.headers["Content-Disposition"]
    assert len(PdfReader(BytesIO(pdf.data)).pages) == 1
    cached = client.post("/figure-builder", data=data)
    assert cached.json == response.json
    assert artifact_path.stat().st_mtime_ns == timestamp
    assert client.get(cached.json["url"]).data == image.data
    assert client.get(cached.json["pdf_url"]).data == pdf.data
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
