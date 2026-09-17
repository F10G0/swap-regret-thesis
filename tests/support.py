import csv

import numpy as np


def read_csv_rows(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def coordination_game_payoffs() -> np.ndarray:
    payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.stack((payoffs, payoffs))
