from algorithms import EliminationAlgorithm


def test_initial_state():
    def phase_length_fn(phase: int) -> int:
        return 2

    algo = EliminationAlgorithm(n_arms=4, phase_length_fn=phase_length_fn)

    assert algo.n_arms == 4
    assert algo.t == 0

    assert algo.phase == 1
    assert algo.phase_t == 0
    assert algo.phase_length == 2

    assert algo.active_arms == [0, 1, 2, 3]
    assert algo.active_index == 0

    assert algo.counts.tolist() == [0, 0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0, 0.0]

    assert algo.phase_counts.tolist() == [0, 0, 0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_select_action_round_robin_within_phase():
    def phase_length_fn(phase: int) -> int:
        return 2

    algo = EliminationAlgorithm(n_arms=3, phase_length_fn=phase_length_fn)

    actions = []
    for _ in range(6):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    # phase 1 length = 2, active arms = [0,1,2]
    # expected round-robin schedule:
    # 0, 1, 2, 0, 1, 2
    assert actions == [0, 1, 2, 0, 1, 2]


def test_global_and_phase_statistics_update_correctly_before_phase_end():
    def phase_length_fn(phase: int) -> int:
        return 3

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    # round 1
    action = algo.select_action()
    assert action == 0
    algo.update(action, reward=1.0)

    assert algo.t == 1
    assert algo.phase_t == 1
    assert algo.counts.tolist() == [1, 0]
    assert algo.reward_sums.tolist() == [1.0, 0.0]
    assert algo.phase_counts.tolist() == [1, 0]
    assert algo.phase_reward_sums.tolist() == [1.0, 0.0]
    assert algo.active_index == 1

    # round 2
    action = algo.select_action()
    assert action == 1
    algo.update(action, reward=0.5)

    assert algo.t == 2
    assert algo.phase_t == 2
    assert algo.counts.tolist() == [1, 1]
    assert algo.reward_sums.tolist() == [1.0, 0.5]
    assert algo.phase_counts.tolist() == [1, 1]
    assert algo.phase_reward_sums.tolist() == [1.0, 0.5]
    assert algo.active_index == 0

    # round 3
    action = algo.select_action()
    assert action == 0
    algo.update(action, reward=0.0)

    assert algo.t == 3
    assert algo.phase_t == 3
    assert algo.counts.tolist() == [2, 1]
    assert algo.reward_sums.tolist() == [1.0, 0.5]
    assert algo.phase_counts.tolist() == [2, 1]
    assert algo.phase_reward_sums.tolist() == [1.0, 0.5]
    assert algo.active_index == 1


def test_phase_transition_resets_phase_local_statistics():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    # phase 1: one pull per arm
    action = algo.select_action()
    assert action == 0
    algo.update(action, reward=1.0)

    action = algo.select_action()
    assert action == 1
    algo.update(action, reward=1.0)

    # after phase 1 ends, algorithm must advance to phase 2
    assert algo.phase == 2
    assert algo.phase_t == 0
    assert algo.active_index == 0

    # global statistics remain
    assert algo.counts.tolist() == [1, 1]
    assert algo.reward_sums.tolist() == [1.0, 1.0]

    # phase-local statistics are reset
    assert algo.phase_counts.tolist() == [0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0]


def test_elimination_rule_keeps_only_arms_within_threshold():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=3, phase_length_fn=phase_length_fn)

    rewards = {
        0: 1.0,
        1: 0.6,
        2: 0.0,
    }

    # phase 1 schedule: 0,1,2
    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    # phase 1 threshold = 2^{-1} = 0.5
    # best empirical mean = 1.0
    # arm 0: 1.0 + 0.5 >= 1.0  keep
    # arm 1: 0.6 + 0.5 >= 1.0  keep
    # arm 2: 0.0 + 0.5 <  1.0  eliminate
    assert algo.active_arms == [0, 1]


def test_elimination_rule_can_reduce_to_single_arm():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=3, phase_length_fn=phase_length_fn)

    rewards = {
        0: 1.0,
        1: 0.0,
        2: 0.0,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    # threshold = 0.5, best mean = 1.0
    # only arm 0 survives
    assert algo.active_arms == [0]
    assert algo.is_commitment_phase()


def test_commitment_phase_always_selects_the_only_remaining_arm():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    rewards = {
        0: 1.0,
        1: 0.0,
    }

    # phase 1: arm 1 should be eliminated
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.active_arms == [0]
    assert algo.is_commitment_phase()

    # after that, select_action should always return 0
    for _ in range(5):
        action = algo.select_action()
        assert action == 0
        algo.update(action, reward=1.0)

    # global stats should keep increasing
    assert algo.counts[0] == 6
    assert algo.counts[1] == 1
    assert algo.t == 7


def test_phase_length_updates_after_advancing_phase():
    def phase_length_fn(phase: int) -> int:
        return phase + 1

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    # initially phase = 1 => phase_length = 2
    assert algo.phase == 1
    assert algo.phase_length == 2

    # phase 1 needs 2 pulls per active arm => total 4 pulls
    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    # after phase transition:
    # phase = 2 => phase_length = 3
    assert algo.phase == 2
    assert algo.phase_length == 3


def test_multiple_phases_with_no_elimination():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    # make both arms always look identical
    rewards = {
        0: 1.0,
        1: 1.0,
    }

    # phase 1
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.active_arms == [0, 1]
    assert algo.phase == 2

    # phase 2
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.active_arms == [0, 1]
    assert algo.phase == 3

    # phase 3
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.active_arms == [0, 1]
    assert algo.phase == 4


def test_empirical_means_uses_global_statistics():
    def phase_length_fn(phase: int) -> int:
        return 2

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    rewards = [1.0, 0.0, 0.0, 1.0]  # actions will be 0,1,0,1
    for reward in rewards:
        action = algo.select_action()
        algo.update(action, reward)

    means = algo.empirical_means()
    assert means[0] == 0.5
    assert means[1] == 0.5


def test_phase_empirical_means_uses_only_current_phase_statistics():
    def phase_length_fn(phase: int) -> int:
        return 1

    algo = EliminationAlgorithm(n_arms=2, phase_length_fn=phase_length_fn)

    # phase 1
    action = algo.select_action()   # 0
    algo.update(action, 1.0)
    action = algo.select_action()   # 1
    algo.update(action, 1.0)

    # phase 2 starts, phase-local stats should be reset
    assert algo.phase == 2
    assert algo.phase_counts.tolist() == [0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0]

    # one action in phase 2
    action = algo.select_action()   # should be 0
    assert action == 0
    algo.update(action, 0.0)

    phase_means = algo.phase_empirical_means()
    assert phase_means[0] == 0.0
    assert phase_means[1] == 0.0


def test_reset_restores_initial_state_completely():
    def phase_length_fn(phase: int) -> int:
        return phase + 1

    algo = EliminationAlgorithm(n_arms=3, phase_length_fn=phase_length_fn)

    rewards = {
        0: 1.0,
        1: 0.5,
        2: 0.0,
    }

    for _ in range(5):
        action = algo.select_action()
        algo.update(action, rewards[action])

    # make sure state has changed
    assert algo.t > 0
    assert algo.phase >= 1
    assert algo.counts.sum() > 0

    algo.reset()

    assert algo.t == 0
    assert algo.phase == 1
    assert algo.phase_t == 0
    assert algo.phase_length == 2

    assert algo.active_arms == [0, 1, 2]
    assert algo.active_index == 0

    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    assert algo.phase_counts.tolist() == [0, 0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0, 0.0]
    