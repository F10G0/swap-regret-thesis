import importlib

import matplotlib as mpl
import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from experiments.algorithm_labels import algorithm_profile_label
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.plots.style import (
    FIGURE_WIDTH, MARKER_STEP, PROFILE_MARKERS, profile_series_style,
    publication_plot, regret_axis_label, regret_series_style, staggered_markevery,
)


@pytest.mark.parametrize("profile,label", [
    (("regret_matching",) * 2, "RM vs RM"),
    (("hedge", "ito_hedge"), "Hedge vs Ito-Hedge"),
    (("auer_exp3", "bm_exp3"), "EXP3 vs BM-EXP3"),
    (("ito_tsallis",), "Ito-Tsallis"),
])
def test_publication_profile_labels(profile, label):
    assert algorithm_profile_label(profile) == label


def test_profile_styles_use_solid_deterministic_marker_cycles_and_phases():
    styles = [profile_series_style(index, 11) for index in range(11)]
    assert [style["marker"] for style in styles] == [*PROFILE_MARKERS, "o"]
    assert {style["linestyle"] for style in styles} == {"-"}
    assert {step for _, step in (style["markevery"] for style in styles)} == {MARKER_STEP}
    assert [style["markevery"][0] for style in styles] == pytest.approx([index / 11 * MARKER_STEP for index in range(11)])
    assert all(0 <= style["markevery"][0] < MARKER_STEP for style in styles)
    assert styles == [profile_series_style(index, 11) for index in range(11)]
    assert profile_series_style(0, 1)["markevery"] == (0, MARKER_STEP)


def test_regret_styles_use_fixed_contrasting_encodings_and_staggered_phases():
    names = ("external", "internal", "swap")
    styles = [regret_series_style(name, index, len(names)) for index, name in enumerate(names)]
    assert [style["color"] for style in styles] == ["#0072B2", "#CC79A7", "#D55E00"]
    assert [style["linestyle"] for style in styles] == ["-", "-.", "--"]
    assert [style["markevery"] for style in styles] == [staggered_markevery(index, 3) for index in range(3)]
    assert styles[2]["zorder"] > styles[0]["zorder"]
    assert styles == [regret_series_style(name, index, len(names)) for index, name in enumerate(names)]


def test_publication_style_is_scoped_even_when_rendering_fails():
    mpl.get_backend()  # Resolve Matplotlib's lazy backend before taking a snapshot.
    before = mpl.rcParams.copy()

    @publication_plot
    def fail():
        assert mpl.rcParams["axes.labelsize"] == 10
        assert mpl.rcParams["pdf.fonttype"] == 42
        raise ValueError("rendering failed")

    with pytest.raises(ValueError, match="rendering failed"):
        fail()
    assert dict(mpl.rcParams) == dict(before)


def fixed_results(directory, bandit=False):
    algorithms = ["auer_exp3", "bm_exp3"] if bandit else ["hedge", "ito_hedge"]
    return [run_cross_play_experiment("rps", [name, name], feedback_mode="bandit" if bandit else "full_information", horizon=100, seed=42,
                   replicate=replicate, output_dir=directory, max_recorded_points=20)
            for name in algorithms for replicate in (0, 1)]


@pytest.mark.parametrize("family", ["regret", "distance", "joint"])
def test_publication_figure_families(tmp_path, monkeypatch, family):
    module_name = {
        "regret": "plot_regret",
        "distance": "plot_equilibrium_convergence",
        "joint": "plot_joint_actions",
    }[family]
    module = importlib.import_module("experiments.plots." + module_name)
    captured = []
    original_save = module.save_figure_pair

    def save(figure, output_path, **kwargs):
        captured.append((figure, output_path))
        return original_save(figure, output_path, **kwargs)

    monkeypatch.setattr(module, "save_figure_pair", save)
    raw = tmp_path / "raw"
    output = tmp_path / (family + ".png")
    if family == "regret":
        rows = [module.load_rows(path) for path in fixed_results(raw)[:2]]
        times, values = module.aggregate_metric_curve(rows, 0, "average_swap_regret")
        curve = module.RegretCurve(times, values, "Hedge vs Hedge", profile_series_style(0, 1))
        module.plot_regret_curves([curve], regret_axis_label("swap"), output)
    else:
        paths = fixed_results(raw)[:2]
        if family == "joint":
            module.plot_joint_actions(paths, output)
        else:
            module.plot_result_equilibrium_distance(paths, output)

    assert len(captured) == 1
    figure, output = captured[0]
    axes = figure.axes[0]
    assert figure.get_figwidth() == FIGURE_WIDTH
    assert 3.8 <= figure.get_figheight() <= 5
    assert axes.get_title() == "" and figure._suptitle is None
    assert axes.xaxis.label.get_fontsize() == axes.yaxis.label.get_fontsize() == 10
    assert all(tick.get_fontsize() == 9 for tick in axes.get_xticklabels())
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for legend in figure.legends:
        box = legend.get_window_extent(renderer)
        assert 0 <= box.x0 < box.x1 <= figure.bbox.width
        assert 0 <= box.y0 < box.y1 < axes.get_window_extent(renderer).y0
        assert all(text.get_fontsize() == 9 for text in legend.get_texts())
    marker_lines = [line for line in axes.lines if line.get_marker() not in {None, "", "None"}]
    if family == "distance":
        assert [line.get_markevery() for line in marker_lines] == [None, None]
    else:
        assert [line.get_markevery() for line in marker_lines] == [
            staggered_markevery(index, len(marker_lines)) for index in range(len(marker_lines))
        ]
    if family == "regret":
        assert axes.get_ylabel() == regret_axis_label("swap")
    pdf = PdfReader(output.with_suffix(".pdf"))
    page = pdf.pages[0]
    assert float(page.mediabox.width) / 72 == pytest.approx(FIGURE_WIDTH)
    assert float(page.mediabox.height) / 72 == pytest.approx(figure.get_figheight())
    assert not list(page.images)  # Lines, heatmap cells, and text are vector PDF artists.
    assert page["/Resources"]["/Font"]
    assert not any(word in page.extract_text().lower() for word in ("expected", "realized", "seed", "solver"))
    with Image.open(output) as preview:
        assert preview.width == round(FIGURE_WIDTH * 150)
        assert abs(preview.height - figure.get_figheight() * 150) <= 1


def test_regret_log_log_plot_uses_mean_tail_and_handles_insufficient_external_points(tmp_path, monkeypatch):
    from experiments.plots import plot_regret as module

    times = np.array([1, 5, 10, 20, 40, 80, 100])
    target = 3 * times.astype(float) ** 0.25
    runs = []
    for factor in (0.5, 1.5):
        values = target * factor
        runs.append([{"player": 0, "t": time, "external_regret": value} for time, value in zip(times, values)])
    mean_times, mean_regret = module.aggregate_metric_curve(runs, 0, "external_regret")
    x, y, fit, message = module.regret_log_log_fit(mean_times, mean_regret, 100)
    retained = np.array([10, 20, 40, 80, 100])
    np.testing.assert_array_equal(x, np.log(retained))
    np.testing.assert_allclose(y, np.log(3 * retained ** 0.25))
    assert fit[0] == pytest.approx(0.25) and message is None

    captured = []
    original_save = module.save_figure_pair

    def save(figure, output_path, **kwargs):
        captured.append(figure)
        return original_save(figure, output_path, **kwargs)

    monkeypatch.setattr(module, "save_figure_pair", save)
    style = regret_series_style("external")
    module.plot_regret_log_log(module.RegretCurve(mean_times, mean_regret,
                               "Replicate-mean cumulative external action regret", style),
                               "external", 100, tmp_path / "fit.png", information_rows=[("View", "Log-log fit")])
    non_positive = module.RegretCurve(np.array([1, 9, 10, 20, 40]), np.array([2, 3, 1, 0, 4]), "", style)
    invalid_x, invalid_y, invalid_fit, invalid_message = module.regret_log_log_fit(
        non_positive.x, non_positive.y, 100)
    assert len(invalid_x) == len(invalid_y) == 0 and invalid_fit is None
    assert invalid_message == module.NON_POSITIVE_LOG_LOG_REGRET
    for metric in ("external", "internal", "swap"):
        curve = module.RegretCurve(non_positive.x, non_positive.y,
                                   f"Replicate-mean cumulative {metric} action regret",
                                   regret_series_style(metric))
        module.plot_regret_log_log(curve, metric, 100, tmp_path / f"non-positive-{metric}.png",
                                   information_rows=[("View", "Log-log fit")])
    insufficient = module.RegretCurve(np.array([1, 20, 40]), np.array([2, 3, 4]),
                                      "Replicate-mean cumulative external action regret", style)
    module.plot_regret_log_log(insufficient, "external", 100, tmp_path / "insufficient.png")

    axes = captured[0].axes[0]
    assert axes.get_xscale() == axes.get_yscale() == "linear"
    assert axes.get_xlabel() == "$\\log t$"
    assert axes.get_ylabel() == "log replicate-mean cumulative external action regret"
    assert "OLS fit (slope = 0.250)" in [line.get_label() for line in axes.lines]
    information = " ".join(PdfReader(tmp_path / "fit.pdf").pages[0].extract_text().split())
    assert "View: Log-log fit" in information and "Fitted slope: 0.250" in information
    for metric, figure in zip(("external", "internal", "swap"), captured[1:4]):
        unavailable_axes = figure.axes[0]
        assert not unavailable_axes.axison
        assert not unavailable_axes.lines
        assert unavailable_axes.get_xlabel() == unavailable_axes.get_ylabel() == ""
        assert [text.get_text() for text in unavailable_axes.texts] == [module.NON_POSITIVE_LOG_LOG_REGRET]
        unavailable = " ".join(PdfReader(tmp_path / f"non-positive-{metric}.pdf").pages[0].extract_text().split())
        assert "Fit: Unavailable" in unavailable and "Fitted slope" not in unavailable
    insufficient_axes = captured[4].axes[0]
    assert not insufficient_axes.axison
    assert not insufficient_axes.lines
    assert insufficient_axes.get_xlabel() == insufficient_axes.get_ylabel() == ""
    assert [text.get_text() for text in insufficient_axes.texts] == [module.INSUFFICIENT_LOG_LOG_POINTS]
    assert (tmp_path / "insufficient.png").is_file() and (tmp_path / "insufficient.pdf").is_file()


def test_horizon_scaling_uses_final_means_log_axes_and_strict_fit_contract(tmp_path, monkeypatch):
    from experiments.plots import plot_regret as module

    runs = [
        [{"horizon": "10", "player": "0", "t": "5", "external_regret": "999"},
         {"horizon": "10", "player": "0", "t": "10", "external_regret": value}]
        for value in ("2", "4")
    ]
    assert module.aggregate_final_metric(runs, 0, "external_regret") == 3

    captured = []
    original_save = module.save_figure_pair
    def save(figure, path, **kwargs):
        captured.append(figure)
        return original_save(figure, path, **kwargs)
    monkeypatch.setattr(module, "save_figure_pair", save)
    horizons = np.array([100, 1_000, 10_000])
    curves = [
        module.RegretCurve(horizons, 2 * horizons ** 0.25, "External regret", regret_series_style("external", 0, 3)),
        module.RegretCurve(horizons, np.array([2.0, 0.0, 4.0]), "Internal regret", regret_series_style("internal", 1, 3)),
        module.RegretCurve(horizons, 3 * horizons ** 0.5, "Swap regret", regret_series_style("swap", 2, 3)),
    ]
    module.plot_horizon_scaling(curves, tmp_path / "scaling.png", information_rows=[("Compare", "Horizons")])
    assert module.horizon_scaling_fit(horizons[:1], curves[0].y[:1]) == (None, module.INSUFFICIENT_HORIZONS)
    assert module.horizon_scaling_fit(horizons[:2], curves[0].y[:2]) == (None, module.INSUFFICIENT_HORIZONS)

    axes = captured[0].axes[0]
    assert axes.get_xscale() == axes.get_yscale() == "log"
    data_lines = [line for line in axes.lines if len(line.get_xdata())]
    assert len(data_lines) == 4
    empirical = [line for line in data_lines if line.get_linestyle() == "None"]
    assert all(line.get_linestyle() == "None" and line.get_markevery() is None for line in empirical)
    assert all(np.array_equal(line.get_xdata(), horizons) for line in empirical)
    assert [line.get_label() for line in data_lines if line.get_linestyle() != "None"] == [
        "External: α = 0.250", "Swap: α = 0.500"]
    assert "Internal: invalid" in [text.get_text() for text in axes.texts]
    information = " ".join(PdfReader(tmp_path / "scaling.pdf").pages[0].extract_text().split())
    assert all(text in information for text in (
        "Compare: Horizons", "External: α = 0.250", "Internal: invalid", "Swap: α = 0.500"))
