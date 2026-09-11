from collections import Counter
import os

import numpy as np
import pytest

import experiments.plots.plot_equilibrium_convergence as plotting
import metrics.equilibrium_distance as metric
from experiments.game_catalog import load_game_payoffs
from experiments.result_trajectories import load_result_action_profiles
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment


def create_result(directory, replicate=0):
    return run_full_information_cross_play_experiment(
        "rps", ["hedge", "bm"], horizon=300, seed=7, replicate=replicate,
        output_dir=directory, max_recorded_points=8,
    )


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
    assert calls == Counter(ce=3, cce=3)
    monkeypatch.setattr(plotting, "load_result_action_profiles", lambda *args: pytest.fail("redraw decoded history"))
    monkeypatch.setattr(plotting, "EQUILIBRIUM_DISTANCE_FIGURE_VERSION", plotting.EQUILIBRIUM_DISTANCE_FIGURE_VERSION + 1)
    plotting.plot_result_equilibrium_distance(path, tmp_path / "redrawn.png", **kwargs)
    assert calls == Counter(ce=3, cce=3)


@pytest.mark.parametrize("count", [1, 3, 159, 160, 161, 2000, 100_000])
def test_distance_points_are_bounded_existing_unique_and_include_endpoints(count):
    horizons = np.arange(1, count + 1) * 3
    indices = plotting.equilibrium_distance_point_indices(horizons)
    selected = horizons[indices]
    assert len(selected) <= 160
    assert selected[0] == horizons[0] and selected[-1] == horizons[-1]
    assert np.all(np.diff(selected) > 0)
    if count <= 160:
        np.testing.assert_array_equal(selected, horizons)


def count_solves(monkeypatch):
    calls = Counter()
    original = metric._PreparedDistanceLP.solve

    def solve(self, vector):
        calls[self.equilibrium] += 1
        return original(self, vector)

    monkeypatch.setattr(metric._PreparedDistanceLP, "solve", solve)
    return calls


def test_first_request_caches_and_unchanged_request_skips_lp_and_history(tmp_path, monkeypatch):
    path = create_result(tmp_path / "raw")
    calls = count_solves(monkeypatch)
    cache = tmp_path / "cache"
    first = plotting._load_result_distances(path, load_game_payoffs("rps"), None, cache)
    assert calls == Counter(ce=3, cce=3)
    assert len(list(cache.glob("*.json"))) == 1
    monkeypatch.setattr(plotting, "load_result_action_profiles", lambda *args: pytest.fail("cache hit decoded history"))
    second = plotting._load_result_distances(path, load_game_payoffs("rps"), None, cache)
    assert calls == Counter(ce=3, cce=3)
    np.testing.assert_array_equal(first.ce, second.ce)
    np.testing.assert_array_equal(first.cce, second.cce)
    np.testing.assert_array_equal(first.horizons, second.horizons)


@pytest.mark.parametrize("change", ["mtime", "size", "payoff", "metric", "format", "policy", "budget", "checkpoints", "corrupt"])
def test_distance_cache_invalidates_only_when_its_inputs_change(tmp_path, monkeypatch, change):
    path = create_result(tmp_path / "raw")
    payoffs = load_game_payoffs("rps")
    cache = tmp_path / "cache"
    calls = count_solves(monkeypatch)
    plotting._load_result_distances(path, payoffs, None, cache)
    initial = sum(calls.values())
    checkpoints = None
    if change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    elif change == "size":
        path.write_text(path.read_text() + "\n")
    elif change == "payoff":
        payoffs = payoffs.copy()
        payoffs[0, 0, 0] += .1
    elif change in {"metric", "format", "policy", "budget"}:
        key = {"metric": "EQUILIBRIUM_DISTANCE_IMPLEMENTATION_VERSION", "format": "DISTANCE_CACHE_VERSION",
               "policy": "DISTANCE_CHECKPOINT_POLICY", "budget": "MAX_EQUILIBRIUM_DISTANCE_POINTS"}[change]
        monkeypatch.setattr(plotting, key, "new-policy" if change == "policy" else 2)
    elif change == "checkpoints":
        checkpoints = np.array([1, 30, 300])
    else:
        next(cache.glob("*.json")).write_text("broken json")
    plotting._load_result_distances(path, payoffs, checkpoints, cache)
    assert sum(calls.values()) > initial


def test_new_replicate_reuses_old_curve_before_aggregating(tmp_path, monkeypatch):
    first = create_result(tmp_path / "raw", replicate=0)
    second = create_result(tmp_path / "raw", replicate=1)
    calls = count_solves(monkeypatch)
    aggregates = []
    monkeypatch.setattr(plotting, "_plot_equilibrium_distance", lambda distances, *args: aggregates.append(distances))
    kwargs = dict(output_path=tmp_path / "figure.png", cache_dir=tmp_path / "cache")
    plotting.plot_result_equilibrium_distance(first, **kwargs)
    assert calls == Counter(ce=3, cce=3)
    plotting.plot_result_equilibrium_distance([first, second], **kwargs)
    assert calls == Counter(ce=6, cce=6)
    plotting.plot_result_equilibrium_distance([second, first], **kwargs)
    assert calls == Counter(ce=6, cce=6)
    assert aggregates[-1].n_replicates == 2
    a = plotting._load_result_distances(first, load_game_payoffs("rps"), None, tmp_path / "cache")
    b = plotting._load_result_distances(second, load_game_payoffs("rps"), None, tmp_path / "cache")
    np.testing.assert_allclose(aggregates[-1].ce_mean, (a.ce + b.ce) / 2)
    np.testing.assert_array_equal(aggregates[-1].ce_mean, aggregates[-2].ce_mean)


def test_selected_horizons_use_exact_compressed_action_history(tmp_path, monkeypatch):
    path = create_result(tmp_path / "raw")
    profiles = load_result_action_profiles(path, (3, 3))
    captured = []
    original = plotting.equilibrium_distance_trajectory

    def capture(payoffs, empirical):
        captured.append(empirical)
        return original(payoffs, empirical)

    monkeypatch.setattr(plotting, "equilibrium_distance_trajectory", capture)
    plotting._load_result_distances(path, load_game_payoffs("rps"), np.arange(1, 301), tmp_path / "cache")
    empirical = captured[0]
    assert len(empirical.horizons) <= 160
    for horizon, vector in zip(empirical.horizons, empirical.vectors):
        counts = np.zeros((3, 3))
        np.add.at(counts, tuple(profiles[:horizon].T), 1)
        np.testing.assert_array_equal(vector, counts.ravel() / horizon)


def test_cleanup_removes_only_distance_figure_files(tmp_path):
    remove = {"foo_equilibrium_distance.png", "foo_equilibrium_distance.pdf",
              "group_replicate_mean_equilibrium_distance.png", "group_replicate_mean_equilibrium_distance.PDF"}
    keep = {"foo_joint_actions_blue_lower_origin.png", "unrelated.png", "foo_equilibrium_distance.csv",
            "foo_equilibrium_distance_detail.png", "payoffs.pdf"}
    for name in remove | keep:
        (tmp_path / name).write_bytes(b"unchanged")
    (tmp_path / "directory_equilibrium_distance.png").mkdir()
    removed = plotting.remove_equilibrium_distance_figures(tmp_path)
    assert {path.name for path in removed} == remove
    for name in keep:
        assert (tmp_path / name).read_bytes() == b"unchanged"
    assert (tmp_path / "directory_equilibrium_distance.png").is_dir()
