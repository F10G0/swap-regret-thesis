import csv
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock

from experiments.result_schema import EXPECTED_REGRET_FIELDNAMES, REALIZED_REGRET_FIELDNAMES
from experiments.results import (
    load_final_result_rows,
    result_algorithm_profile,
    result_game_payoff_digest,
    result_player_algorithm,
    result_regret_evaluation,
)


SUMMARY_REGRET_FIELDS = tuple(field for field in EXPECTED_REGRET_FIELDNAMES + REALIZED_REGRET_FIELDNAMES if field.startswith("average_"))


@dataclass(frozen=True)
class ResultSnapshot:
    filenames: list[str]
    summaries: list[dict]
    warnings: list[str]


class ResultIndex:
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self._cache: dict[tuple[str, int, int], tuple[list[dict], str | None]] = {}
        self._lock = Lock()

    def snapshot(self) -> ResultSnapshot:
        with self._lock:
            paths = sorted(self.raw_dir.glob("*.csv")) if self.raw_dir.exists() else []
            filenames = []
            summaries = []
            warnings = []
            active_keys = set()

            for path in paths:
                try:
                    stat = path.stat()
                except OSError:
                    continue

                filenames.append(path.name)
                cache_key = (str(path.absolute()), stat.st_mtime_ns, stat.st_size)
                active_keys.add(cache_key)
                cached = self._cache.get(cache_key)
                if cached is None:
                    cached = self._summarize_file(path)
                    self._cache[cache_key] = cached

                file_summaries, warning = cached
                summaries.extend(file_summaries)
                if warning:
                    warnings.append(warning)

            self._cache = {key: value for key, value in self._cache.items() if key in active_keys}
            return ResultSnapshot(filenames, summaries, warnings)

    def _summarize_file(self, path: Path) -> tuple[list[dict], str | None]:
        try:
            rows = load_final_result_rows(path)
            if not rows:
                raise ValueError("file has no result rows")
            n_players = len(result_algorithm_profile(rows[0]))
            players = sorted(int(row["player"]) for row in rows)
            if players != list(range(n_players)):
                raise ValueError("file has incomplete final player rows")

            summaries = []
            for row in sorted(rows, key=lambda result: int(result["player"])):
                player = int(row["player"])
                if int(row["t"]) != int(row["horizon"]):
                    raise ValueError(f"player {player} has no final-horizon row")
                summaries.append(self._summary_from_row(path.name, row))
            return summaries, None
        except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
            return [], f"Skipped {path.name}: {error}"

    def _summary_from_row(self, filename: str, row: dict[str, str]) -> dict:
        algorithm_profile = result_algorithm_profile(row)
        player = int(row["player"])
        result = {
            "experiment": filename,
            "game": row["game"],
            "run_id": row["run_id"],
            "feedback_mode": row["feedback_mode"],
            "regret_evaluation": result_regret_evaluation(row),
            "seed": int(row["seed"]),
            "replicate": int(row["replicate"]),
            "stationary_method": row["stationary_method"],
            "game_payoff_digest": result_game_payoff_digest(row),
            "player": player,
            "n_players": len(algorithm_profile),
            "algorithm_profile": list(algorithm_profile),
            "player_algorithm": result_player_algorithm(row),
            "co_player_algorithms": [algorithm for position, algorithm in enumerate(algorithm_profile) if position != player],
            "algorithm_player_0": algorithm_profile[0],
            "algorithm_player_1": algorithm_profile[1],
            "horizon": int(row["horizon"]),
        }
        for field in SUMMARY_REGRET_FIELDS:
            if field not in row:
                continue
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite value for {field}")
            result[field] = value
        return result
