import ast
import importlib
from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from experiments.plots.style import (
    ALGORITHM_STYLES, FIGURE_WIDTH, algorithm_style, curve_labels,
    profile_label, publication_plot, regret_axis_label,
)
from experiments.scenarios.adversarial import (
    ALGORITHMS_BY_FEEDBACK_MODE, RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment,
)
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from experiments.scenarios.bandit_cross_play import run_bandit_cross_play_experiment
from experiments.scenarios.adversarial_scaling import AdversarialScalingSpec, run_adversarial_scaling_experiment


@pytest.mark.parametrize("profile,label", [
    (("ito", "ito"), "Ito"), (("hedge", "hedge"), "Hedge"),
    (("stationary_regret_matching",) * 2, "SRM"),
    (("hedge", "ito"), "Hedge vs Ito"),
])
def test_publication_profile_labels(profile, label):
    assert profile_label(profile) == label


def test_style_mapping_is_global_order_independent_and_redundant():
    algorithms = set().union(*ALGORITHMS_BY_FEEDBACK_MODE.values())
    assert algorithms <= ALGORITHM_STYLES.keys()
    forward = {name: algorithm_style(name) for name in sorted(algorithms)}
    assert forward == {name: algorithm_style(name) for name in sorted(algorithms, reverse=True)}
    assert len({(s["linestyle"], s["marker"]) for s in forward.values()}) == len(algorithms)
    altered = algorithm_style("ito")
    altered["color"] = "red"
    assert algorithm_style("ito")["color"] != "red"


def test_labels_include_only_metadata_needed_to_distinguish_curves():
    base = dict(algorithm="hedge_vs_hedge", seed=42, stationary_method="solve", horizon=100)
    assert curve_labels([base, base | {"algorithm": "ito_vs_ito"}]) == ["Hedge", "Ito"]
    assert curve_labels([base, base | {"seed": 7}]) == ["Hedge · seed 42", "Hedge · seed 7"]
    assert curve_labels([base, base | {"stationary_method": "pinv"}]) == ["Hedge · solver solve", "Hedge · solver pinv"]


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
    runner = run_bandit_cross_play_experiment if bandit else run_full_information_cross_play_experiment
    algorithms = ["auer_exp3", "bm"] if bandit else ["hedge", "ito"]
    return [runner("rps", [name, name], horizon=100, seed=42,
                   replicate=replicate, output_dir=directory, max_recorded_points=20)
            for name in algorithms for replicate in (0, 1)]


@pytest.mark.parametrize("family", ["rps", "bandit", "ilrw", "scaling", "distance", "joint", "weights"])
def test_publication_figure_families(tmp_path, monkeypatch, family):
    module_name = {
        "rps": "plot_regret", "bandit": "plot_regret", "ilrw": "plot_adversarial",
        "scaling": "plot_adversarial_scaling", "distance": "plot_equilibrium_convergence",
        "joint": "plot_joint_actions", "weights": "plot_equilibrium_weights",
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
    if family in {"rps", "bandit"}:
        fixed_results(raw, family == "bandit")
        groups = module.group_replicate_runs(module.collect_results(raw)["rps"])
        module.plot_regret("rps", groups, "swap", 0, True, tmp_path)
    elif family == "ilrw":
        for name in ("auer_exp3", "bm", "ito", "exp3_ix", "lce_ix"):
            for replicate in (0, 1):
                run_adversarial_experiment(name, n_actions=9, horizon=100, seed=42,
                    feedback_mode="bandit", environment=RANDOM_WALK_ENVIRONMENT,
                    environment_seed=7, replicate=replicate, output_dir=raw)
        module._plot_regret(module.collect_adversarial_results(raw), RANDOM_WALK_ENVIRONMENT,
                            "bandit", 9, "swap", False, output)
    elif family == "scaling":
        spec = AdversarialScalingSpec(environment=RANDOM_WALK_ENVIRONMENT,
            feedback_mode="bandit", algorithm_name="auer_exp3", action_counts=(3, 6, 9),
            replicates=2, horizon=100, environment_seed=7, learner_seed=42)
        path = run_adversarial_scaling_experiment(spec, raw, workers=1)
        module._plot_scaling(module.load_adversarial_scaling_rows(path), output)
    elif family == "weights":
        from experiments.game_catalog import load_game_payoffs
        module.plot_equilibrium_profile_weights(load_game_payoffs("rps"), "ce", output)
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
    for line in axes.lines:
        if line.get_label() in {"AuerExp3", "BM", "Ito", "Hedge", "EXP3-IX", "LCE-IX"}:
            name = {"AuerExp3": "auer_exp3", "BM": "bm", "Ito": "ito", "Hedge": "hedge",
                    "EXP3-IX": "exp3_ix", "LCE-IX": "lce_ix"}[line.get_label()]
            expected = algorithm_style(name)
            assert (line.get_color(), line.get_linestyle(), line.get_marker()) == (
                expected["color"], expected["linestyle"], expected["marker"])
    if family in {"rps", "bandit", "ilrw", "scaling"}:
        view = "final" if family == "scaling" else "sqrt_scaling" if family == "ilrw" else "average"
        kind = "external" if family == "scaling" else "swap"
        assert axes.get_ylabel() == regret_axis_label(kind, view)
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


def test_every_figure_renderer_uses_the_shared_publication_style():
    paths = list(Path("experiments/plots").glob("plot_*.py"))
    renderers = []
    for path in paths:
        for function in ast.walk(ast.parse(path.read_text())):
            if not isinstance(function, ast.FunctionDef):
                continue
            creates_figure = any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "plt"
                and node.func.attr in {"figure", "subplots"}
                for node in ast.walk(function)
            )
            if creates_figure:
                renderers.append(function.name)
                assert any(isinstance(node, ast.Name) and node.id == "publication_plot"
                           for node in function.decorator_list), f"{path}:{function.lineno}"
    assert len(renderers) == 6
