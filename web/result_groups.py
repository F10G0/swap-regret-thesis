from collections import defaultdict
from hashlib import sha256
import json

import numpy as np

from metrics.confidence import mean_confidence_interval_half_width
from web.result_index import SUMMARY_REGRET_FIELDS


RESULT_GROUP_FIELDS = (
    "game",
    "game_payoff_digest",
    "feedback_mode",
    "regret_evaluation",
    "horizon",
    "seed",
    "stationary_method",
)


def result_group_key(summary: dict) -> tuple:
    return (
        *(summary.get(field, "") for field in RESULT_GROUP_FIELDS),
        tuple(summary["algorithm_profile"]),
    )


def result_group_id(key: tuple) -> str:
    payload = json.dumps(key, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _confidence_interval(values: list[float]) -> float:
    return float(mean_confidence_interval_half_width(values))


def _replicate_label(replicates: list[int]) -> str:
    if len(replicates) == 1:
        return str(replicates[0])
    if replicates == list(range(replicates[0], replicates[-1] + 1)):
        return f"{replicates[0]}–{replicates[-1]}"
    return ", ".join(map(str, replicates))


def aggregate_result_summaries(summaries: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[result_group_key(summary)].append(summary)

    aggregated = []
    for key, group_rows in grouped.items():
        group_id = result_group_id(key)
        rows_by_player = defaultdict(list)
        for row in group_rows:
            rows_by_player[row["player"]].append(row)

        for player in sorted(rows_by_player):
            rows_by_replicate = {}
            for row in rows_by_player[player]:
                rows_by_replicate.setdefault(row["replicate"], row)
            rows = [rows_by_replicate[replicate] for replicate in sorted(rows_by_replicate)]
            replicates = [row["replicate"] for row in rows]
            confidence_intervals = {}
            result = dict(rows[0])
            for field in SUMMARY_REGRET_FIELDS:
                values = [row[field] for row in rows if field in row]
                if len(values) != len(rows):
                    continue
                result[field] = float(np.mean(values))
                confidence_intervals[field] = _confidence_interval(values)

            result.update({
                "group_id": group_id,
                "replicate": replicates[0],
                "replicates": replicates,
                "replicate_count": len(replicates),
                "replicate_label": _replicate_label(replicates),
                "confidence_intervals": confidence_intervals,
                "runs": rows,
            })
            aggregated.append(result)
    return aggregated


def result_group_filenames(summaries: list[dict], group_id: str) -> list[str]:
    filenames = {
        summary["experiment"]
        for summary in summaries
        if result_group_id(result_group_key(summary)) == group_id
    }
    if not filenames:
        raise KeyError(group_id)
    return sorted(filenames)
