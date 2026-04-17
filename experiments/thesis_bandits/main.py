from algorithms import ExploreThenCommit


def test_round_robin_exploration():
    algo = ExploreThenCommit(n_arms=3, m=2)

    actions = []
    for _ in range(6):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2, 0, 1, 2]


def test_commit_after_exploration():
    algo = ExploreThenCommit(n_arms=3, m=1)

    rewards = {
        0: 0.1,
        1: 0.2,
        2: 0.9,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    next_action = algo.select_action()
    assert next_action == 2


def test_reset():
    algo = ExploreThenCommit(n_arms=3, m=2)

    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.committed_arm is None
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    