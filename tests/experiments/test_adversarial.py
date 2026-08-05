import csv

import pytest

from experiments.plots.plot_adversarial import plot_adversarial_results
from experiments.runner import ExperimentCancelled
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    adversarial_memory_label,
    adversarial_memory_window,
    load_adversarial_rows,
    run_adversarial_experiment,
)


def _rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_adversarial_experiment_records_both_regret_sources(tmp_path) -> None:
    output_path = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert len(rows) == 5
    assert {row["environment"] for row in rows} == {
        "historical_frequency_v2"
    }
    assert {row["memory"] for row in rows} == {"full_history"}
    assert "replicate" not in rows[0]
    assert rows[0]["punished_action"] == "0"
    assert rows[-1]["t"] == "5"
    assert "average_expected_swap_regret" in rows[0]
    assert "average_realized_swap_regret" in rows[0]
    assert load_adversarial_rows(output_path)[-1] == rows[-1]


def test_adversarial_seeds_have_distinct_identity_and_streams(tmp_path) -> None:
    first = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=20,
        seed=7,
        output_dir=tmp_path,
    )
    second = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=20,
        seed=8,
        output_dir=tmp_path,
    )

    first_rows = _rows(first)
    second_rows = _rows(second)

    assert first != second
    assert [row["action"] for row in first_rows] != [
        row["action"] for row in second_rows
    ]


def test_adversarial_experiment_is_atomic_on_cancellation(tmp_path) -> None:
    with pytest.raises(ExperimentCancelled):
        run_adversarial_experiment(
            "hedge",
            horizon=3,
            output_dir=tmp_path,
            should_cancel=lambda: True,
        )

    assert list(tmp_path.iterdir()) == []


def test_adversarial_plotter_creates_average_and_scaled_regret_figures(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    for seed in (7, 8):
        run_adversarial_experiment(
            "hedge",
            n_actions=3,
            horizon=5,
            seed=seed,
            output_dir=raw_dir,
        )

    generated = plot_adversarial_results(raw_dir, figure_dir)

    expected_regret_figures = {
        f"historical_frequency_3_actions_average_{source}_{regret}_regret.png"
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    expected_scaling_figures = {
        f"historical_frequency_3_actions_{source}_{regret}_regret_over_sqrt_t.png"
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    expected_behavior_figures = {
        f"{path.stem}_{kind}_frequency.png"
        for path in raw_dir.glob("*.csv")
        for kind in ("action", "punished_action")
    }
    assert len(generated) == 16
    assert {path.name for path in generated} == (
        expected_regret_figures
        | expected_scaling_figures
        | expected_behavior_figures
    )
    assert all(path.with_suffix(".pdf").is_file() for path in generated)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_actions": 1}, "between 2"),
        ({"horizon": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"memory_window": -1}, "non-negative"),
    ],
)
def test_adversarial_spec_validation(changes, message) -> None:
    values = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
        "memory_window": 0,
    }
    with pytest.raises(ValueError, match=message):
        AdversarialExperimentSpec(**(values | changes))


def test_adversarial_memory_is_part_of_run_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }

    assert AdversarialExperimentSpec(
        **common,
        memory_window=10,
    ).run_id != AdversarialExperimentSpec(
        **common,
        memory_window=0,
    ).run_id


def test_arbitrary_adversarial_memory_window() -> None:
    assert adversarial_memory_window("full_history") == 0
    assert adversarial_memory_window("last_37") == 37
    assert adversarial_memory_label(37) == "Last 37 rounds"


def test_unknown_adversarial_memory_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown adversarial memory"):
        adversarial_memory_window("unknown")
