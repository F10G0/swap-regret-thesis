import numpy as np

from metrics.regret import ExpectedRegretBundle, RealizedRegretBundle, RegretBundles


def test_expected_replacement_regret_matches_hand_calculation() -> None:
    bundle = ExpectedRegretBundle(n_actions=2)

    bundle.update(np.array([0.25, 0.75]), np.array([0.0, 1.0]))

    assert bundle.external_regret == 0.25
    assert bundle.internal_regret == 0.25
    assert bundle.swap_regret == 0.25


def test_realized_replacement_regret_matches_hand_calculation() -> None:
    bundle = RealizedRegretBundle(n_actions=2)

    bundle.update(0, np.array([0.0, 1.0]))

    assert bundle.external_regret == 1.0
    assert bundle.internal_regret == 1.0
    assert bundle.swap_regret == 1.0


def test_regret_definitions_are_distinct() -> None:
    bundle = ExpectedRegretBundle(n_actions=3)
    bundle.cumulative_replacement_gains = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 1.0], [4.0, -2.0, 0.0]])

    assert bundle.external_regret == 7.0
    assert bundle.internal_regret == 4.0
    assert bundle.swap_regret == 9.0


def test_expected_replacement_gains_accumulate_across_rounds() -> None:
    bundle = ExpectedRegretBundle(n_actions=2)

    bundle.update(np.array([0.25, 0.75]), np.array([0.0, 1.0]))
    bundle.update(np.array([0.5, 0.5]), np.array([1.0, 0.0]))

    assert np.allclose(bundle.cumulative_replacement_gains, [[0.0, -0.25], [-0.25, 0.0]])


def test_summary_reports_cumulative_and_average_regret() -> None:
    bundle = ExpectedRegretBundle(n_actions=3)
    bundle.cumulative_replacement_gains = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 1.0], [4.0, -2.0, 0.0]])

    assert bundle.summary(time=2) == {
        "expected_external_regret": 7.0,
        "average_expected_external_regret": 3.5,
        "expected_internal_regret": 4.0,
        "average_expected_internal_regret": 2.0,
        "expected_swap_regret": 9.0,
        "average_expected_swap_regret": 4.5,
    }


def test_regret_bundles_update_expected_and_realized_regret() -> None:
    bundles = RegretBundles(n_actions=2)

    bundles.update(np.array([0.5, 0.5]), action=0, payoff_vector=np.array([0.0, 1.0]))

    assert bundles.expected.external_regret == 0.5
    assert bundles.realized.external_regret == 1.0
