import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.plots import confidence_free_figure_path
from experiments.plots.plot_adversarial_scaling import (
    aggregate_scaling_regret,
    plot_adversarial_scaling_results,
)
from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT
from experiments.scenarios.adversarial_scaling import (
    AdversarialScalingSpec,
    load_adversarial_scaling_rows,
    run_adversarial_scaling_experiment,
)


def scaling_spec(**overrides) -> AdversarialScalingSpec:
    values = {
        "environment": RANDOM_WALK_ENVIRONMENT,
        "initialization_mode": "centered",
        "feedback_mode": "bandit",
        "algorithm_name": "exp3",
        "action_counts": (2, 4),
        "replicates": 3,
        "horizon": 4,
        "environment_seed": 11,
        "learner_seed": 23,
    }
    return AdversarialScalingSpec(**(values | overrides))


def test_scaling_experiment_uses_common_seed_schedule_at_every_action_count(
    tmp_path: Path,
) -> None:
    output_path = run_adversarial_scaling_experiment(scaling_spec(), tmp_path)
    rows = load_adversarial_scaling_rows(output_path)

    assert [(int(row["n_actions"]), int(row["replicate"])) for row in rows] == [
        (2, 0),
        (2, 1),
        (2, 2),
        (4, 0),
        (4, 1),
        (4, 2),
    ]
    for replicate in range(3):
        matched = [row for row in rows if int(row["replicate"]) == replicate]
        assert {row["learner_seed"] for row in matched} == {str(23 + replicate)}
        assert {row["environment_seed"] for row in matched} == {str(11 + replicate)}
    assert {row["target_regret"] for row in rows} == {"external"}
    assert all(np.isfinite(float(row["expected_regret"])) for row in rows)
    assert all(np.isfinite(float(row["realized_regret"])) for row in rows)
    assert list(tmp_path.glob("*.csv")) == [output_path]


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_counts": (2,)},
        {"action_counts": (2, 2)},
        {"action_counts": (1, 2)},
        {"replicates": 0},
        {"regret_evaluation": "unknown"},
    ],
)
def test_scaling_spec_rejects_invalid_batches(overrides: dict) -> None:
    with pytest.raises(ValueError):
        scaling_spec(**overrides)


def test_scaling_aggregation_uses_student_t_confidence_intervals() -> None:
    rows = [
        {"n_actions": "2", "expected_regret": "1", "realized_regret": "2"},
        {"n_actions": "2", "expected_regret": "3", "realized_regret": "4"},
        {"n_actions": "4", "expected_regret": "2", "realized_regret": "3"},
        {"n_actions": "4", "expected_regret": "6", "realized_regret": "7"},
    ]

    action_counts, means, confidence = aggregate_scaling_regret(rows, "expected")

    assert action_counts.tolist() == [2, 4]
    assert means.tolist() == [2, 4]
    assert confidence == pytest.approx([12.706204736, 25.412409472])


def test_scaling_plot_writes_expected_and_realized_figure_pairs(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    run_adversarial_scaling_experiment(scaling_spec(replicates=2), raw_dir)

    generated = plot_adversarial_scaling_results(raw_dir, figure_dir)

    assert {path.name for path in generated} == {
        f"{scaling_spec(replicates=2).run_id}_expected_regret_by_actions.png",
        f"{scaling_spec(replicates=2).run_id}_realized_regret_by_actions.png",
    }
    assert all(path.is_file() and path.with_suffix(".pdf").is_file() for path in generated)
    assert all(
        confidence_free_figure_path(path).is_file()
        and confidence_free_figure_path(path).with_suffix(".pdf").is_file()
        for path in generated
    )


def test_scaling_experiment_respects_selected_regret_evaluation(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    spec = scaling_spec(replicates=2, regret_evaluation="expected")
    output_path = run_adversarial_scaling_experiment(spec, raw_dir)

    rows = load_adversarial_scaling_rows(output_path)
    generated = plot_adversarial_scaling_results(raw_dir, figure_dir)

    assert {row["regret_evaluation"] for row in rows} == {"expected"}
    assert all(row["expected_regret"] for row in rows)
    assert all(not row["realized_regret"] for row in rows)
    assert [path.name for path in generated] == [
        f"{spec.run_id}_expected_regret_by_actions.png"
    ]


def test_scaling_loader_treats_legacy_results_as_both(tmp_path: Path) -> None:
    generated = run_adversarial_scaling_experiment(
        scaling_spec(replicates=2),
        tmp_path,
    )
    with generated.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    fieldnames = [field for field in rows[0] if field != "regret_evaluation"]
    legacy_path = tmp_path / "legacy.csv"
    with legacy_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                field: value
                for field, value in row.items()
                if field != "regret_evaluation"
            }
            for row in rows
        )

    loaded = load_adversarial_scaling_rows(legacy_path)

    assert {row["regret_evaluation"] for row in loaded} == {"both"}
