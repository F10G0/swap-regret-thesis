from dataclasses import replace
from pathlib import Path

import pytest

from experiments.result_catalog import FixedDetails, ResultRecord, ResultSet, SUMMARY_REGRET_FIELDS


def record(replicate, regrets, **changes):
    source = ResultRecord(
        path=Path(f"run-{replicate}.csv"), kind="fixed", run_id=f"run-{replicate}", feedback_mode="bandit",
        horizon=100, runtime_environment="{}", runtime_fingerprint="",
        profile=("exp3_ix", "exp3_ix"),
        details=FixedDetails("rps", "", 42, replicate, "solve", ("exp3_ix", "exp3_ix")),
        final_values=tuple({"average_external_regret": value} for value in regrets),
    )
    return replace(source, **changes)


def test_all_supported_regret_metrics_have_replicate_means():
    records = tuple(record(replicate, (), final_values=tuple(
        {field: float(index + 2 * replicate) for index, field in enumerate(SUMMARY_REGRET_FIELDS)}
        for _ in range(2))) for replicate in (0, 1))
    result = ResultSet(records).summaries(grouped=True)[0]
    for index, field in enumerate(SUMMARY_REGRET_FIELDS):
        assert result[field] == pytest.approx(index + 1.0)


@pytest.mark.parametrize("field,first_value,second_value", [
    ("seed", 42, 43), ("game_payoff_digest", "a" * 64, "b" * 64),
])
def test_incompatible_result_metadata_is_not_combined(field, first_value, second_value):
    records = []
    for replicate, value in enumerate((first_value, second_value)):
        source = record(replicate, (0.1, 0.2))
        records.append(replace(source, details=source.details._replace(**{field: value})))
    groups = ResultSet(tuple(records)).groups("dashboard")
    assert len(groups) == 2 and all(len(group.records) == 1 for group in groups)


def test_duplicate_replicates_keep_first_summary_but_all_detail_sources():
    results = ResultSet((record(4, (0.3, 0.4)), record(3, (0.1, 0.2)),
                         record(3, (99.0, 99.0), path=Path("duplicate.csv"))))
    groups = results.summaries(grouped=True)
    assert [row["player"] for row in groups] == [0, 1]
    assert groups[0]["group_id"] == groups[1]["group_id"]
    assert groups[0]["replicates"] == [3, 4]
    assert groups[0]["replicate_label"] == "3–4"
    assert groups[0]["average_external_regret"] == 0.2
    assert groups[1]["average_external_regret"] == pytest.approx(0.3)
    assert [run["replicate"] for run in groups[0]["runs"]] == [3, 4]
    assert [run["experiment"] for run in groups[0]["runs"]] == ["run-3.csv", "run-4.csv"]
    assert [p.name for p in results.detail_paths(groups[0]["group_id"])] == ["duplicate.csv", "run-3.csv", "run-4.csv"]
    assert [p.name for p in results.canonical_detail_paths(groups[0]["group_id"])] == ["run-3.csv", "run-4.csv"]
    with pytest.raises(KeyError):
        results.canonical_detail_paths("missing")
