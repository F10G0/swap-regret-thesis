import csv
from pathlib import Path

import numpy as np
import pytest

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
from experiments.result_schema import REGRET_FIELDNAMES
from experiments.seeding import (
    ENVIRONMENT_SEED_DOMAIN,
    LEARNER_SEED_DOMAIN,
    domain_separated_seed,
)


def scaling_spec(**overrides) -> AdversarialScalingSpec:
    values = {
        "environment": RANDOM_WALK_ENVIRONMENT,
        "feedback_mode": "bandit",
        "algorithm_name": "exp3_ix",
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
        assert {row["learner_seed"] for row in matched} == {
            str(domain_separated_seed(23, replicate, LEARNER_SEED_DOMAIN))
        }
        assert {row["environment_seed"] for row in matched} == {
            str(domain_separated_seed(11, replicate, ENVIRONMENT_SEED_DOMAIN))
        }
    assert {row["target_regret"] for row in rows} == {"external"}
    assert all(np.isfinite(float(row[field])) for row in rows for field in REGRET_FIELDNAMES)
    assert list(tmp_path.glob("*.csv")) == [output_path]


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_counts": (2,)},
        {"action_counts": (2, 2)},
        {"action_counts": (1, 2)},
        {"replicates": 0},
    ],
)
def test_scaling_spec_rejects_invalid_batches(overrides: dict) -> None:
    with pytest.raises(ValueError):
        scaling_spec(**overrides)


def test_scaling_aggregation_uses_replicate_means() -> None:
    rows = [
        {"n_actions": "2", "external_regret": "1", "target_regret": "external"},
        {"n_actions": "2", "external_regret": "3", "target_regret": "external"},
        {"n_actions": "4", "external_regret": "2", "target_regret": "external"},
        {"n_actions": "4", "external_regret": "6", "target_regret": "external"},
    ]

    action_counts, means = aggregate_scaling_regret(rows)

    assert action_counts.tolist() == [2, 4]
    assert means.tolist() == [2, 4]


def test_scaling_plot_writes_canonical_figure_pairs(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    run_adversarial_scaling_experiment(scaling_spec(replicates=2), raw_dir)

    generated = plot_adversarial_scaling_results(raw_dir, figure_dir)

    assert {path.name for path in generated} == {
        f"{scaling_spec(replicates=2).run_id}_regret_by_actions.png",
    }
    assert all(path.is_file() and path.with_suffix(".pdf").is_file() for path in generated)
    assert len(list(figure_dir.glob("*.png"))) == len(generated)
    assert len(list(figure_dir.glob("*.pdf"))) == len(generated)


@pytest.mark.parametrize("version", [None, 2, 3, 4])
def test_scaling_loader_rejects_stale_results_without_modifying_them(tmp_path, version):
    generated = run_adversarial_scaling_experiment(scaling_spec(replicates=2), tmp_path)
    with generated.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if version is None:
            row.pop("implementation_version")
        else:
            row["implementation_version"] = str(version)
    with generated.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = generated.read_bytes()
    with pytest.raises(ValueError, match="incompatible result implementation_version"):
        load_adversarial_scaling_rows(generated)
    assert generated.read_bytes() == before
