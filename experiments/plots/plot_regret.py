import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURE_DIR, RAW_DIR


REGRET_COLUMNS = {
    "external": "external_regret",
    "internal": "internal_regret",
    "swap": "swap_regret",
}


def load_rows(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)

    with input_path.open("r", newline="") as file:
        return list(csv.DictReader(file))


def collect_results(
    input_dir: str | Path = RAW_DIR,
) -> dict[str, dict[str, list[dict]]]:
    input_dir = Path(input_dir)
    results = defaultdict(lambda: defaultdict(list))

    for path in sorted(input_dir.glob("*.csv")):
        rows = load_rows(path)

        if not rows:
            continue

        first_row = rows[0]
        if "game" not in first_row or "algorithm" not in first_row:
            print(f"[skip] incompatible csv: {path}")
            continue

        game_name = first_row["game"]
        algorithm_name = first_row["algorithm"]

        results[game_name][algorithm_name].extend(rows)

    return {
        game_name: dict(rows_by_algorithm)
        for game_name, rows_by_algorithm in results.items()
    }


def available_players(rows_by_algorithm: dict[str, list[dict]]) -> list[int]:
    return sorted(
        {
            int(row["player"])
            for rows in rows_by_algorithm.values()
            for row in rows
        }
    )


def rows_for_player(rows: list[dict], player: int) -> list[dict]:
    return [
        row
        for row in rows
        if int(row["player"]) == player
    ]


def plot_average_regret(
    game_name: str,
    rows_by_algorithm: dict[str, list[dict]],
    regret_name: str,
    player: int,
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    average_column = f"average_{REGRET_COLUMNS[regret_name]}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))

    for algorithm_name, rows in rows_by_algorithm.items():
        player_rows = rows_for_player(rows, player)

        t_values = [
            int(row["t"])
            for row in player_rows
        ]
        regret_values = [
            float(row[average_column])
            for row in player_rows
        ]

        plt.plot(
            t_values,
            regret_values,
            label=algorithm_name,
        )

    plt.xlabel("Round")
    plt.ylabel(f"Average {regret_name} regret")
    plt.title(f"{game_name}: average {regret_name} regret, player {player}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{game_name}_average_{regret_name}_regret_player_{player}.png"
    )
    plt.close()


def plot_regret_over_sqrt_t(
    game_name: str,
    rows_by_algorithm: dict[str, list[dict]],
    regret_name: str,
    player: int,
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    regret_column = REGRET_COLUMNS[regret_name]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))

    for algorithm_name, rows in rows_by_algorithm.items():
        player_rows = rows_for_player(rows, player)

        t_values = [
            int(row["t"])
            for row in player_rows
        ]
        regret_values = [
            float(row[regret_column]) / np.sqrt(int(row["t"]))
            for row in player_rows
        ]

        plt.plot(
            t_values,
            regret_values,
            label=algorithm_name,
        )

    plt.xscale("log")
    plt.xlabel("Round")
    plt.ylabel(f"{regret_name.capitalize()} regret / sqrt(t)")
    plt.title(f"{game_name}: {regret_name} regret scaling, player {player}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{game_name}_{regret_name}_regret_over_sqrt_t_player_{player}.png"
    )
    plt.close()


def plot_all_results(
    input_dir: str | Path = RAW_DIR,
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    results = collect_results(input_dir)

    for game_name, rows_by_algorithm in results.items():
        players = available_players(rows_by_algorithm)

        for player in players:
            for regret_name in REGRET_COLUMNS:
                plot_average_regret(
                    game_name=game_name,
                    rows_by_algorithm=rows_by_algorithm,
                    regret_name=regret_name,
                    player=player,
                    output_dir=output_dir,
                )

                plot_regret_over_sqrt_t(
                    game_name=game_name,
                    rows_by_algorithm=rows_by_algorithm,
                    regret_name=regret_name,
                    player=player,
                    output_dir=output_dir,
                )


def main() -> None:
    plot_all_results()


if __name__ == "__main__":
    main()
