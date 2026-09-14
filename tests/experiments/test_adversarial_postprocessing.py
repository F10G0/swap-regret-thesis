import csv
import json
import os

import pytest

import experiments.scenarios.adversarial as scenario
import experiments.plots.plot_adversarial as plots
from experiments.scenarios.cross_play import run_cross_play_experiment
from tests.support import read_csv_rows


def create_run(directory, **kwargs):
    return scenario.run_adversarial_experiment(
        "auer_exp3", feedback_mode="bandit", environment=scenario.RANDOM_WALK_ENVIRONMENT,
        horizon=20, output_dir=directory, **kwargs,
    )


def rewrite_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_constant_metadata_is_validated_once_but_every_observation_is_checked(tmp_path, monkeypatch):
    path = create_run(tmp_path)
    metadata_calls, observations = [], []
    validate_metadata = scenario._validate_adversarial_metadata
    validate_observation = scenario._validate_adversarial_observation

    def metadata(row, path):
        metadata_calls.append(row["t"])
        return validate_metadata(row, path)

    def observation(row, path, n_actions):
        observations.append(row["t"])
        return validate_observation(row, path, n_actions)

    monkeypatch.setattr(scenario, "_validate_adversarial_metadata", metadata)
    monkeypatch.setattr(scenario, "_validate_adversarial_observation", observation)
    expected = read_csv_rows(path)
    assert scenario.load_adversarial_rows(path) == expected
    assert metadata_calls == ["1"]
    assert observations == [str(t) for t in range(1, 21)]


@pytest.mark.parametrize("field", scenario.ADVERSARIAL_IDENTITY_FIELDS)
def test_metadata_changes_on_unplotted_rows_are_rejected(tmp_path, field):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    rows[1][field] = "corrupted"
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="inconsistent metadata"):
        scenario.load_adversarial_rows(path, max_points=2)


@pytest.mark.parametrize("field,value", [("action", "100"), ("current_best_action", "100"), ("current_best_reward", "nan")])
def test_invalid_observations_on_unplotted_rows_are_rejected(tmp_path, field, value):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    rows[1][field] = value
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="invalid"):
        scenario.load_adversarial_rows(path, max_points=2)


def test_scoped_collection_never_loads_unrelated_trajectories(tmp_path, monkeypatch):
    wanted = create_run(tmp_path, n_actions=7)
    create_run(tmp_path, n_actions=5)
    create_run(tmp_path, n_actions=9)
    scenario.run_adversarial_experiment("hedge", n_actions=7, horizon=20, output_dir=tmp_path)
    # Scope discovery must not depend on the generated filename.
    renamed = tmp_path / "renamed.csv"
    wanted.rename(renamed)
    loaded = []
    original = plots.load_adversarial_rows

    def load(path, **kwargs):
        assert path == renamed
        loaded.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(plots, "load_adversarial_rows", load)
    result = plots.collect_adversarial_results(tmp_path, scope=(scenario.RANDOM_WALK_ENVIRONMENT, "bandit", 7))
    assert [path for path, _ in result] == loaded == [renamed]


def test_unchanged_trajectories_use_cache_and_deleted_results_disappear(tmp_path, monkeypatch):
    path = create_run(tmp_path / "raw")
    first = plots.collect_adversarial_results(path.parent)

    def unexpected_load(*args, **kwargs):
        pytest.fail("unchanged trajectory was loaded again")

    monkeypatch.setattr(plots, "load_adversarial_rows", unexpected_load)
    assert plots.collect_adversarial_results(path.parent) == first
    path.unlink()
    assert plots.collect_adversarial_results(path.parent) == []


@pytest.mark.parametrize("change", ["source", "cache_json", "cache_rows", "cache_version", "sampling"])
def test_cache_invalidates_when_source_or_format_changes(tmp_path, monkeypatch, change):
    path = create_run(tmp_path / "raw")
    cache = tmp_path / "cache"
    plots.collect_adversarial_results(path.parent, cache_dir=cache)
    cache_path = cache / f"{path.stem}.json"
    if change == "source":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    elif change == "cache_json":
        cache_path.write_text("invalid json")
    elif change == "cache_rows":
        payload = json.loads(cache_path.read_text())
        payload["rows"] = [{}]
        cache_path.write_text(json.dumps(payload))
    elif change == "cache_version":
        monkeypatch.setattr(plots, "PLOT_ROW_CACHE_VERSION", plots.PLOT_ROW_CACHE_VERSION + 1)
    else:
        monkeypatch.setattr(plots, "MAX_PLOT_POINTS", 2)
    calls = []
    original = plots.load_adversarial_rows

    def load(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(plots, "load_adversarial_rows", load)
    plots.collect_adversarial_results(path.parent, cache_dir=cache)
    assert calls == [path]


def test_cached_file_that_becomes_invalid_is_not_silently_reused(tmp_path):
    path = create_run(tmp_path)
    plots.collect_adversarial_results(tmp_path)
    rows = read_csv_rows(path)
    rows[1]["runtime_fingerprint"] = "invalid"
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="inconsistent metadata"):
        plots.collect_adversarial_results(tmp_path)
    assert plots.collect_adversarial_results(tmp_path, skip_invalid=True) == []


def test_scoped_plot_generation_preserves_other_figures(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    output = tmp_path / "figures"
    output.mkdir()
    create_run(raw, n_actions=7)
    create_run(raw, n_actions=5)
    scope = (scenario.RANDOM_WALK_ENVIRONMENT, "bandit", 7)
    prefix = plots.adversarial_figure_prefix(scope)
    unrelated = output / "adversarial_unrelated.png"
    unrelated.write_bytes(b"preserve")
    stale = output / f"{prefix}obsolete.pdf"
    stale.write_bytes(b"stale")

    def plot(results, environment, feedback_mode, n_actions, regret_name, average, output_path):
        assert (environment, feedback_mode, n_actions) == scope
        assert all(int(rows[0]["n_actions"]) == 7 for _, rows in results)
        output_path.write_bytes(b"new")
        output_path.with_suffix(".pdf").write_bytes(b"new")

    monkeypatch.setattr(plots, "_plot_regret", plot)
    generated = plots.plot_adversarial_results(raw, output, scope=scope)
    assert len(generated) == 6
    assert unrelated.read_bytes() == b"preserve"
    assert not stale.exists()


def test_mutation_during_row_loading_preserves_format_specific_policy(tmp_path, monkeypatch):
    from experiments.plots import plot_regret as fixed

    fixed_path = run_cross_play_experiment("rps", ["hedge", "hedge"], horizon=2, output_dir=tmp_path, feedback_mode="full_information")
    adversarial_path = create_run(tmp_path / "adversarial")

    def mutate(path):
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    original_fixed, original_adversarial = fixed.iter_result_rows, plots.load_adversarial_rows

    def fixed_rows(path):
        yield from original_fixed(path)
        mutate(path)

    def adversarial_rows(path, **kwargs):
        rows = original_adversarial(path, **kwargs)
        mutate(path)
        return rows

    monkeypatch.setattr(fixed, "iter_result_rows", fixed_rows)
    monkeypatch.setattr(plots, "load_adversarial_rows", adversarial_rows)
    assert fixed.load_rows(fixed_path, cache_dir=tmp_path / "fixed-cache")
    assert not list((tmp_path / "fixed-cache").glob("*.json"))
    with pytest.raises(ValueError, match="changed while its trajectory"):
        plots._load_plot_rows(adversarial_path, tmp_path / "adversarial-cache")
