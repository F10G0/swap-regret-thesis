from algorithms.external_regret import Hedge
from algorithms.swap_regret import FullBM, FullIto
from config import HORIZON, RAW_DIR, SEED
from environments import RepeatedGame
from experiments.games import PAYOFF_FACTORIES
from experiments.recorder import CsvRecorder
from experiments.runner import run_game


LEARNING_RATE = 0.3

ALGORITHMS = {
    "hedge": Hedge,
    "bm": FullBM,
    "ito": FullIto,
}

REGRET_FIELDNAMES = [
    "game",
    "algorithm",
    "t",
    "player",
    "action",
    "payoff",
    "external_regret",
    "average_external_regret",
    "internal_regret",
    "average_internal_regret",
    "swap_regret",
    "average_swap_regret",
]


def create_players(
    algorithm_class,
    n_players: int,
    n_actions: int,
):
    return [
        algorithm_class(
            n_actions=n_actions,
            learning_rate=LEARNING_RATE,
            seed=SEED + player_id,
        )
        for player_id in range(n_players)
    ]


def main() -> None:
    for game_name, payoff_factory in PAYOFF_FACTORIES.items():
        payoffs = payoff_factory()
        game = RepeatedGame(payoffs)

        n_players = game.n_players
        n_actions = game.n_actions[0]

        for algorithm_name, algorithm_class in ALGORITHMS.items():
            recorder = CsvRecorder(REGRET_FIELDNAMES)

            players = create_players(
                algorithm_class=algorithm_class,
                n_players=n_players,
                n_actions=n_actions,
            )

            run_game(
                game_name=game_name,
                algorithm_name=algorithm_name,
                game=game,
                players=players,
                recorder=recorder,
                horizon=HORIZON,
            )

            recorder.save(
                RAW_DIR / f"{game_name}_{algorithm_name}.csv"
            )

            print(f"[done] {game_name} - {algorithm_name}")


if __name__ == "__main__":
    main()
