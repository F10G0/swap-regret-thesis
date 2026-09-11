import numpy as np

import pytest

from metrics.regret import RegretBundle


def test_strategy_weighted_replacement_regret_matches_hand_calculation() -> None:
    bundle = RegretBundle(n_actions=2)

    bundle.update(np.array([0.25, 0.75]), np.array([0.0, 1.0]))

    assert bundle.external_regret == 0.25
    assert bundle.internal_regret == 0.25
    assert bundle.swap_regret == 0.25


def test_regret_definitions_are_distinct() -> None:
    bundle = RegretBundle(n_actions=3)
    bundle.cumulative_replacement_gains = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 1.0], [4.0, -2.0, 0.0]])

    assert bundle.external_regret == 7.0
    assert bundle.internal_regret == 4.0
    assert bundle.swap_regret == 9.0


def test_strategy_weighted_replacement_gains_accumulate_across_rounds() -> None:
    bundle = RegretBundle(n_actions=2)

    bundle.update(np.array([0.25, 0.75]), np.array([0.0, 1.0]))
    bundle.update(np.array([0.5, 0.5]), np.array([1.0, 0.0]))

    assert np.allclose(bundle.cumulative_replacement_gains, [[0.0, -0.25], [-0.25, 0.0]])


def test_summary_reports_cumulative_and_average_regret() -> None:
    bundle = RegretBundle(n_actions=3)
    bundle.cumulative_replacement_gains = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 1.0], [4.0, -2.0, 0.0]])

    assert bundle.summary(time=2) == {
        "external_regret": 7.0,
        "average_external_regret": 3.5,
        "internal_regret": 4.0,
        "average_internal_regret": 2.0,
        "swap_regret": 9.0,
        "average_swap_regret": 4.5,
    }


def test_matrix_update_matches_strategy_weighted_formula() -> None:
    rng = np.random.default_rng(42)
    bundle = RegretBundle(9)
    matrix = np.zeros((9, 9))
    for _ in range(30):
        strategy = rng.dirichlet(np.ones(9))
        payoffs = rng.random(9)
        matrix += strategy[:, None] * (payoffs[None, :] - payoffs[:, None])
        bundle.update(strategy, payoffs)
        np.testing.assert_array_equal(bundle.cumulative_replacement_gains, matrix)


@pytest.mark.parametrize("n_actions", [0, -1])
def test_tracker_requires_positive_action_count(n_actions) -> None:
    with pytest.raises(ValueError, match="n_actions"):
        RegretBundle(n_actions)


def test_per_replicate_maximum_is_not_maximum_of_mean_matrix() -> None:
    from experiments.plots.plot_regret import aggregate_metric_curve

    trajectories = []
    for strategy, payoffs in [
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.0, 1.0], [1.0, 0.0]),
    ]:
        bundle = RegretBundle(2)
        bundle.update(np.asarray(strategy), np.asarray(payoffs))
        trajectories.append([{"t": "1", "player": "0", **bundle.summary(1)}])
    for name in ("external", "internal", "swap"):
        _, means = aggregate_metric_curve(trajectories, 0, f"{name}_regret")
        assert means.tolist() == [1.0]
    # Averaging these gain matrices first would instead give external/internal 0.5.
