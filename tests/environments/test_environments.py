import numpy as np
import pytest

from algorithms.external_regret import Hedge
from environments import (
    BanditRepeatedGame,
    HistoricalFrequencyAdversary,
    RepeatedGame,
)
from experiments.runner import run_game


def asymmetric_payoff_tensor() -> np.ndarray:
    return np.array(
        [
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            [
                [0.6, 0.5, 0.4],
                [0.3, 0.2, 0.1],
            ],
        ]
    )


def test_game_dimensions_are_derived_from_payoff_tensor() -> None:
    game = RepeatedGame(asymmetric_payoff_tensor())

    assert game.n_players == 2
    assert game.n_actions == (2, 3)


def test_full_information_feedback_computes_payoff_vectors_on_demand() -> None:
    game = RepeatedGame(asymmetric_payoff_tensor())
    actions = (1, 2)
    game.step(actions)

    first_feedback = game.feedback(0)
    assert np.array_equal(first_feedback, [0.3, 0.6])
    assert np.array_equal(game.feedback(1), [0.3, 0.2, 0.1])
    assert [game.feedback(player)[action] for player, action in enumerate(actions)] == [
        0.6,
        0.1,
    ]
    assert game.feedback(0) is not first_feedback
    assert not hasattr(game, "_payoff_vectors")


def test_feedback_models_reveal_only_permitted_information() -> None:
    payoff_tensor = asymmetric_payoff_tensor()
    full_information_game = RepeatedGame(payoff_tensor)
    bandit_game = BanditRepeatedGame(payoff_tensor)

    full_information_game.step((1, 2))
    bandit_game.step((1, 2))

    assert np.array_equal(full_information_game.feedback(0), [0.3, 0.6])
    assert np.array_equal(full_information_game.feedback(1), [0.3, 0.2, 0.1])
    assert bandit_game.feedback(0) == pytest.approx(0.6)
    assert bandit_game.feedback(1) == pytest.approx(0.1)


def test_deviation_payoffs_are_computed_on_demand_for_evaluation() -> None:
    game = BanditRepeatedGame(asymmetric_payoff_tensor())
    game.step((1, 2))

    first_payoffs = game.deviation_payoffs(0)

    assert np.array_equal(first_payoffs, [0.3, 0.6])
    assert np.array_equal(game.deviation_payoffs(1), [0.3, 0.2, 0.1])
    assert game.deviation_payoffs(0) is not first_payoffs
    assert not hasattr(game, "_deviation_payoffs")


def test_feedback_is_independent_from_payoff_tensor() -> None:
    payoff_tensor = asymmetric_payoff_tensor()
    game = RepeatedGame(payoff_tensor)
    game.step((1, 2))

    payoff_tensor[0, 1, 2] = 0.0
    assert game.feedback(0)[1] == pytest.approx(0.6)

    feedback = game.feedback(0)
    feedback[0] = 0.0
    assert game.feedback(0)[0] == pytest.approx(0.3)


@pytest.mark.parametrize("player", [-1, 2])
def test_feedback_rejects_invalid_player_indices(player: int) -> None:
    game = RepeatedGame(asymmetric_payoff_tensor())
    game.step((1, 2))
    with pytest.raises(IndexError, match="invalid player index"):
        game.feedback(player)


@pytest.mark.parametrize("actions", [(-1, 2), (2, 2), (1, -1), (1, 3)])
def test_step_rejects_invalid_action_indices(actions: tuple[int, ...]) -> None:
    game = RepeatedGame(asymmetric_payoff_tensor())
    with pytest.raises(IndexError, match="invalid action index"):
        game.step(actions)


def test_step_rejects_wrong_number_of_actions() -> None:
    game = RepeatedGame(asymmetric_payoff_tensor())
    with pytest.raises(ValueError, match="number of actions"):
        game.step((1,))


@pytest.mark.parametrize(
    "payoff_tensor, message",
    [
        (np.array([0.0, 1.0]), "must have shape"),
        (np.empty((0, 0)), "number of players"),
        (np.full((2, 2, 2), np.nan), "finite"),
        (np.full((2, 2, 2), 1.1), r"\[0, 1\]"),
        (np.empty((2, 0, 2)), "at least one action"),
        (np.zeros((2, 2, 2, 2)), "number of players"),
    ],
)
def test_environment_rejects_invalid_payoff_tensor(
    payoff_tensor: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RepeatedGame(payoff_tensor)


def test_historical_frequency_adversary_uses_only_past_actions() -> None:
    environment = HistoricalFrequencyAdversary(3)

    environment.step((2,))

    np.testing.assert_array_equal(environment.feedback(), [0.0, 1.0, 1.0])
    assert environment.punished_action == 0
    np.testing.assert_array_equal(environment.action_counts, [0, 0, 1])

    environment.step((1,))

    np.testing.assert_array_equal(environment.feedback(), [1.0, 1.0, 0.0])
    assert environment.punished_action == 2
    np.testing.assert_array_equal(environment.action_counts, [0, 1, 1])


def test_historical_frequency_payoff_does_not_depend_on_current_action() -> None:
    first = HistoricalFrequencyAdversary(3)
    second = HistoricalFrequencyAdversary(3)
    for environment in (first, second):
        environment.step((1,))
        environment.step((1,))

    first.step((0,))
    second.step((2,))

    np.testing.assert_array_equal(
        first.deviation_payoffs(),
        second.deviation_payoffs(),
    )
    assert first.punished_action == second.punished_action == 1


def test_historical_frequency_adversary_rotates_ties() -> None:
    environment = HistoricalFrequencyAdversary(3)

    punished_actions = []
    for action in (0, 1, 2, 0):
        environment.step((action,))
        punished_actions.append(environment.punished_action)

    assert punished_actions == [0, 0, 1, 2]


def test_historical_frequency_adversary_can_use_a_finite_window() -> None:
    environment = HistoricalFrequencyAdversary(3, memory_window=1)

    punished_actions = []
    for action in (2, 1, 0):
        environment.step((action,))
        punished_actions.append(environment.punished_action)

    assert punished_actions == [0, 2, 1]
    np.testing.assert_array_equal(environment.action_counts, [1, 1, 1])


def test_runner_steps_environment_once_per_round() -> None:
    class CountingGame(RepeatedGame):
        def __init__(self, payoff_tensor: np.ndarray) -> None:
            super().__init__(payoff_tensor)
            self.step_count = 0

        def step(self, actions: tuple[int, ...]) -> None:
            self.step_count += 1
            super().step(actions)

    class MemoryRecorder:
        def __init__(self) -> None:
            self.rows = []

        def record(self, row: dict) -> None:
            self.rows.append(row)

    game = CountingGame(asymmetric_payoff_tensor())
    players = [
        Hedge(2, horizon=3, seed=0),
        Hedge(3, horizon=3, seed=1),
    ]
    recorder = MemoryRecorder()

    run_game(
        game_name="asymmetric",
        feedback_mode="full_information",
        algorithm_name="hedge_vs_hedge",
        game=game,
        players=players,
        recorder=recorder,
        horizon=3,
    )

    assert game.step_count == 3
    assert len(recorder.rows) == 6
