from algorithms import UpperConfidenceBound


def test_initialization_phase():
    algo = UpperConfidenceBound(n_arms=3)

    actions = []
    for _ in range(3):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2]


def test_select_best_ucb_after_initialization():
    algo = UpperConfidenceBound(n_arms=3, exploration_factor=2.0)

    # initialization phase: one pull per arm
    rewards = {
        0: 1.0,
        1: 0.0,
        2: 0.0,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    # empirical means:
    # arm 0: 1.0
    # arm 1: 0.0
    # arm 2: 0.0
    #
    # counts are all 1, so bonuses are equal.
    # therefore the arm with largest empirical mean should be chosen.
    next_action = algo.select_action()
    assert next_action == 0


def test_reset():
    algo = UpperConfidenceBound(n_arms=3)

    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    