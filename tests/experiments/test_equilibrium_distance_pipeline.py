from collections import Counter
import os

import numpy as np
import pytest

import experiments.plots.plot_equilibrium_convergence as plotting
import metrics.equilibrium_distance as metric
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.game_catalog import load_game_payoffs
from experiments.recording import joint_action_histogram_checkpoints
from experiments.result_trajectories import load_result_empirical_distribution_trajectory


def create_result(directory, replicate=0):
    return run_cross_play_experiment(
        "rps", ["hedge", "bm_hedge"], horizon=300, seed=7, replicate=replicate,
        output_dir=directory, max_recorded_points=8,
        feedback_mode="full_information",
    )


DISTANCE_POINT_COUNT = len(joint_action_histogram_checkpoints(300))


@pytest.mark.parametrize("replicate_count", [1, 2])
def test_distance_figures_use_caption_free_publication_layout(tmp_path, monkeypatch, replicate_count):
    from pypdf import PdfReader

    paths = [create_result(tmp_path / "raw", replicate=index) for index in range(replicate_count)]
    saved = []
    original_save = plotting.save_figure_pair

    def capture(figure, output_path):
        axes = figure.axes[0]
        saved.append((figure._suptitle, axes.get_title(), axes.get_legend_handles_labels()[1]))
        original_save(figure, output_path)

    monkeypatch.setattr(plotting, "save_figure_pair", capture)
    output = tmp_path / "distance.png"
    plotting.plot_result_equilibrium_distance(paths, output)

    title, annotation, legend = saved[0]
    assert title is None
    assert annotation == ""
    assert legend == ["CE", "CCE"]
    assert output.is_file()
    pdf_text = PdfReader(output.with_suffix(".pdf")).pages[0].extract_text()
    assert "seed" not in pdf_text and "solver" not in pdf_text
    assert "CE" in pdf_text and "CCE" in pdf_text


def test_style_redraw_reuses_cached_distance_values(tmp_path, monkeypatch):
    path = create_result(tmp_path / "raw")
    calls = count_solves(monkeypatch)
    kwargs = dict(cache_dir=tmp_path / "cache")
    plotting.plot_result_equilibrium_distance(path, tmp_path / "first.png", **kwargs)
    assert calls == Counter(ce=DISTANCE_POINT_COUNT, cce=DISTANCE_POINT_COUNT)
    monkeypatch.setattr(plotting, "load_result_empirical_distribution_trajectory", lambda *args: pytest.fail("redraw loaded histograms"))
    monkeypatch.setattr(plotting, "EQUILIBRIUM_DISTANCE_FIGURE_VERSION", plotting.EQUILIBRIUM_DISTANCE_FIGURE_VERSION + 1)
    plotting.plot_result_equilibrium_distance(path, tmp_path / "redrawn.png", **kwargs)
    assert calls == Counter(ce=DISTANCE_POINT_COUNT, cce=DISTANCE_POINT_COUNT)


def count_solves(monkeypatch):
    calls = Counter()
    original = metric._PreparedDistanceLP.solve

    def solve(self, vector):
        calls[self.equilibrium] += 1
        return original(self, vector)

    monkeypatch.setattr(metric._PreparedDistanceLP, "solve", solve)
    return calls


def test_first_request_caches_and_unchanged_request_skips_lp_and_histograms(tmp_path, monkeypatch):
    path = create_result(tmp_path / "raw")
    calls = count_solves(monkeypatch)
    cache = tmp_path / "cache"
    first = plotting._load_result_distances(path, load_game_payoffs("rps"), cache)
    assert calls == Counter(ce=DISTANCE_POINT_COUNT, cce=DISTANCE_POINT_COUNT)
    assert len(list(cache.glob("*.json"))) == 1
    monkeypatch.setattr(plotting, "load_result_empirical_distribution_trajectory", lambda *args: pytest.fail("cache hit loaded histograms"))
    second = plotting._load_result_distances(path, load_game_payoffs("rps"), cache)
    assert calls == Counter(ce=DISTANCE_POINT_COUNT, cce=DISTANCE_POINT_COUNT)
    np.testing.assert_array_equal(first.ce, second.ce)
    np.testing.assert_array_equal(first.cce, second.cce)
    np.testing.assert_array_equal(first.horizons, second.horizons)


@pytest.mark.parametrize("change", ["mtime", "size", "payoff", "metric", "format", "corrupt"])
def test_distance_cache_invalidates_only_when_its_inputs_change(tmp_path, monkeypatch, change):
    path = create_result(tmp_path / "raw")
    payoffs = load_game_payoffs("rps")
    cache = tmp_path / "cache"
    calls = count_solves(monkeypatch)
    plotting._load_result_distances(path, payoffs, cache)
    initial = sum(calls.values())
    if change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    elif change == "size":
        path.write_text(path.read_text() + "\n")
    elif change == "payoff":
        payoffs = payoffs.copy()
        payoffs[0, 0, 0] += .1
    elif change in {"metric", "format"}:
        key = {"metric": "EQUILIBRIUM_DISTANCE_IMPLEMENTATION_VERSION", "format": "DISTANCE_CACHE_VERSION"}[change]
        monkeypatch.setattr(plotting, key, 3)
    else:
        next(cache.glob("*.json")).write_text("broken json")
    plotting._load_result_distances(path, payoffs, cache)
    assert sum(calls.values()) > initial


def test_new_replicate_reuses_old_curve_before_aggregating(tmp_path, monkeypatch):
    first = create_result(tmp_path / "raw", replicate=0)
    second = create_result(tmp_path / "raw", replicate=1)
    calls = count_solves(monkeypatch)
    aggregates = []
    monkeypatch.setattr(plotting, "_plot_equilibrium_distance", lambda distances, *args: aggregates.append(distances))
    kwargs = dict(output_path=tmp_path / "figure.png", cache_dir=tmp_path / "cache")
    plotting.plot_result_equilibrium_distance(first, **kwargs)
    assert calls == Counter(ce=DISTANCE_POINT_COUNT, cce=DISTANCE_POINT_COUNT)
    plotting.plot_result_equilibrium_distance([first, second], **kwargs)
    assert calls == Counter(ce=2 * DISTANCE_POINT_COUNT, cce=2 * DISTANCE_POINT_COUNT)
    plotting.plot_result_equilibrium_distance([second, first], **kwargs)
    assert calls == Counter(ce=2 * DISTANCE_POINT_COUNT, cce=2 * DISTANCE_POINT_COUNT)
    assert aggregates[-1].n_replicates == 2
    a = plotting._load_result_distances(first, load_game_payoffs("rps"), tmp_path / "cache")
    b = plotting._load_result_distances(second, load_game_payoffs("rps"), tmp_path / "cache")
    np.testing.assert_allclose(aggregates[-1].ce_mean, (a.ce + b.ce) / 2)
    np.testing.assert_array_equal(aggregates[-1].ce_mean, aggregates[-2].ce_mean)


def test_distances_use_every_stored_histogram_checkpoint(tmp_path, monkeypatch):
    path = create_result(tmp_path / "raw")
    stored = load_result_empirical_distribution_trajectory(path, (3, 3))
    captured = []
    original = plotting.equilibrium_distance_trajectory

    def capture(payoffs, empirical):
        captured.append(empirical)
        return original(payoffs, empirical)

    monkeypatch.setattr(plotting, "equilibrium_distance_trajectory", capture)
    plotting._load_result_distances(path, load_game_payoffs("rps"), tmp_path / "cache")
    empirical = captured[0]
    np.testing.assert_array_equal(empirical.horizons, joint_action_histogram_checkpoints(300))
    np.testing.assert_array_equal(empirical.horizons, stored.horizons)
    np.testing.assert_array_equal(empirical.vectors, stored.vectors)
