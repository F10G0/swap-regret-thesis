import pytest

from web.result_groups import aggregate_result_summaries, result_group_filenames


def summary(replicate: int, player: int, regret: float) -> dict:
    return {
        "experiment": f"run-{replicate}.csv",
        "run_id": f"run-{replicate}",
        "game": "rps",
        "feedback_mode": "bandit",
        "regret_evaluation": "realized",
        "seed": 42,
        "replicate": replicate,
        "stationary_method": "solve",
        "player": player,
        "n_players": 2,
        "algorithm_profile": ["exp3", "exp3"],
        "player_algorithm": "exp3",
        "co_player_algorithms": ["exp3"],
        "algorithm_player_0": "exp3",
        "algorithm_player_1": "exp3",
        "horizon": 100,
        "average_realized_external_regret": regret,
    }


def test_result_summaries_are_aggregated_by_replicate_and_player() -> None:
    summaries = [
        summary(3, 0, 0.1),
        summary(3, 1, 0.2),
        summary(4, 0, 0.3),
        summary(4, 1, 0.4),
    ]

    aggregated = aggregate_result_summaries(summaries)

    assert len(aggregated) == 2
    assert aggregated[0]["replicates"] == [3, 4]
    assert aggregated[0]["replicate_label"] == "3–4"
    assert aggregated[0]["average_realized_external_regret"] == pytest.approx(0.2)
    assert aggregated[0]["confidence_intervals"]["average_realized_external_regret"] == pytest.approx(1.2706204736)
    assert [run["replicate"] for run in aggregated[0]["runs"]] == [3, 4]
    assert aggregated[0]["group_id"] == aggregated[1]["group_id"]
    assert result_group_filenames(summaries, aggregated[0]["group_id"]) == ["run-3.csv", "run-4.csv"]


def test_different_base_seeds_are_not_combined() -> None:
    first = summary(0, 0, 0.1)
    second = summary(1, 0, 0.2)
    second["seed"] = 43

    aggregated = aggregate_result_summaries([first, second])

    assert len(aggregated) == 2
    assert all(row["replicate_count"] == 1 for row in aggregated)


def test_different_regret_evaluations_are_not_combined() -> None:
    first = summary(0, 0, 0.1)
    second = summary(1, 0, 0.2)
    second["regret_evaluation"] = "both"
    second["average_expected_external_regret"] = 0.3

    aggregated = aggregate_result_summaries([first, second])

    assert len(aggregated) == 2
    assert {row["regret_evaluation"] for row in aggregated} == {"realized", "both"}


def test_different_payoff_tensors_are_not_combined() -> None:
    first = summary(0, 0, 0.1)
    second = summary(1, 0, 0.2)
    first["game_payoff_digest"] = "a" * 64
    second["game_payoff_digest"] = "b" * 64

    aggregated = aggregate_result_summaries([first, second])

    assert len(aggregated) == 2
    assert all(row["replicate_count"] == 1 for row in aggregated)
