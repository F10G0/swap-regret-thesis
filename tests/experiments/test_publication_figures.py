import importlib

import matplotlib as mpl
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
