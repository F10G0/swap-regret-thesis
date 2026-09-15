"""Persisted result sources and explicit consumer-specific membership policies.

Only final observations are read here. Histogram trajectories continue to use
the strict reader; presentation dictionaries never feed grouping.
"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
from threading import Lock
from typing import Literal, NamedTuple

import numpy as np

from experiments.algorithm_labels import algorithm_label
from experiments.result_schema import REGRET_NAMES
from experiments.results import load_final_result_rows, result_algorithm_profile
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS, TARGET_REGRET_BY_ALGORITHM,
    adversarial_environment_detail, load_final_adversarial_row,
)
from experiments.scenarios.adversarial_scaling import load_adversarial_scaling_rows, adversarial_scaling_environment_detail
from experiments.scenarios.cross_play import FEEDBACK_MODE_LABELS


ResultKind = Literal["fixed", "adversarial", "scaling"]
SUMMARY_REGRET_FIELDS = tuple(f"average_{name}_regret" for name in REGRET_NAMES)
FIXED_COMPARISON_FIELDS = ("game", "game_payoff_digest", "feedback_mode", "horizon", "seed",
                           "stationary_method", "runtime_fingerprint")


def adversarial_plot_key(row: dict[str, str]) -> tuple:
    # Unlike builder compatibility, this key includes reward_step, retains wire
    # strings, and determines plot sorting. Do not substitute comparison_key.
    return (row["environment"], row["reward_step"], row["feedback_mode"], row["runtime_fingerprint"],
            row["n_actions"], row["algorithm"], row["horizon"],
            int(row["base_environment_seed"]) if row["environment_seed"] else None, int(row["base_learner_seed"]))


class FixedDetails(NamedTuple):
    game: str
    game_payoff_digest: str
    seed: int
    replicate: int
    stationary_method: str
    player_algorithms: tuple[str, ...]


class AdversarialDetails(NamedTuple):
    environment: str
    reward_step: str
    base_environment_seed: int | None
    environment_seed: int | None
    base_learner_seed: int
    learner_seed: int
    replicate: int
    n_actions: int


class ScalingDetails(NamedTuple):
    environment: str
    base_environment_seed: int | None
    base_learner_seed: int
    action_counts: tuple[int, ...]
    replicates: int
    target_regret: str


@dataclass(frozen=True)
class ResultRecord:
    path: Path
    kind: ResultKind
    run_id: str
    feedback_mode: str
    horizon: int
    runtime_environment: str
    runtime_fingerprint: str
    profile: tuple[str, ...]
    details: FixedDetails | AdversarialDetails | ScalingDetails
    # Fixed: player order; adversarial: one final row. Scaling exposes metadata
    # only; its strict CSV reader still validates the complete final-result grid.
    final_values: tuple[dict[str, float], ...]

    @classmethod
    def read(cls, path: Path, kind: ResultKind) -> "ResultRecord":
        if kind == "fixed":
            rows = sorted(load_final_result_rows(path), key=lambda row: int(row["player"]))
        elif kind == "adversarial":
            rows = [load_final_adversarial_row(path)]
        elif kind == "scaling":
            rows = load_adversarial_scaling_rows(path)
        else:
            raise ValueError(f"unknown result kind: {kind}")
        if not rows:
            raise ValueError("file has no result rows")
        row = rows[0]
        if kind == "fixed":
            profile = result_algorithm_profile(row)
            player_algorithms = []
            for observation, fallback in zip(rows, profile):
                override = observation.get("player_algorithm", "").strip()
                player_algorithms.append(fallback if override in ("", "0") else override)
            details = FixedDetails(row["game"], row.get("game_payoff_digest", "").strip(), int(row["seed"]),
                                   int(row["replicate"]), row["stationary_method"], tuple(player_algorithms))
            if details.game_payoff_digest == "0":
                details = details._replace(game_payoff_digest="")
        else:
            profile = (row["algorithm"],)
            environment_seed = int(row["base_environment_seed"]) if row["base_environment_seed"] else None
            if kind == "adversarial":
                details = AdversarialDetails(row["environment"], row["reward_step"], environment_seed,
                    int(row["environment_seed"]) if row["environment_seed"] else None,
                    int(row["base_learner_seed"]), int(row["learner_seed"]), int(row["replicate"]), int(row["n_actions"]))
            else:
                details = ScalingDetails(row["environment"], environment_seed, int(row["base_learner_seed"]),
                    tuple(map(int, row["action_counts"].split(","))), int(row["replicates"]), row["target_regret"])
        values = () if kind == "scaling" else tuple({field: float(value) for field, value in observation.items()
                        if field in SUMMARY_REGRET_FIELDS}
                       for observation in rows)
        if kind == "fixed":
            for observation in values:
                for field, value in observation.items():
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite value for {field}")
        # Fixed summaries canonicalize runtime JSON; one-player readers validate it
        # but leave its original serialization intact.
        runtime = row["runtime_environment"]
        if kind == "fixed":
            runtime = json.dumps(json.loads(runtime), sort_keys=True, separators=(",", ":"))
        fingerprint = row["runtime_fingerprint"].strip()
        return cls(path, kind, row["run_id"], row["feedback_mode"], int(row["horizon"]),
                   runtime, fingerprint, profile, details, values)

    @property
    def scope(self) -> str:
        return self.details.game if isinstance(self.details, FixedDetails) else self.details.environment

    @property
    def replicate(self) -> int:
        if isinstance(self.details, ScalingDetails):
            raise ValueError("a scaling source is an aggregate, not one replicate")
        return self.details.replicate

    def metrics(self, player: int = 0) -> list[str]:
        return [name for name in REGRET_NAMES if f"average_{name}_regret" in self.final_values[player]]

    @property
    def comparison_key(self) -> tuple:
        info = self.details
        if isinstance(info, FixedDetails):
            return tuple(getattr(self, field) if hasattr(self, field) else getattr(info, field)
                         for field in FIXED_COMPARISON_FIELDS)
        if isinstance(info, AdversarialDetails):
            return (info.environment, self.feedback_mode, info.n_actions, self.horizon, info.base_learner_seed,
                    info.base_environment_seed, self.runtime_fingerprint)
        raise ValueError("scaling aggregates do not define trajectory comparison groups")

    @property
    def group_id(self) -> str:
        return sha256(json.dumps((*self.comparison_key, self.profile), separators=(",", ":")).encode()).hexdigest()[:16]

    def summary(self, player: int = 0) -> dict:
        """Serialize a final observation for presentation, never for grouping."""
        info = self.details
        common = {name: getattr(self, name) for name in ("feedback_mode", "horizon")}
        if isinstance(info, FixedDetails):
            return common | {"experiment": self.path.name, "run_id": self.run_id, "game": info.game,
                "seed": info.seed, "replicate": info.replicate, "stationary_method": info.stationary_method,
                "game_payoff_digest": info.game_payoff_digest, "runtime_environment": self.runtime_environment,
                "runtime_fingerprint": self.runtime_fingerprint, "player": player, "n_players": len(self.profile),
                "algorithm_profile": list(self.profile), "player_algorithm": info.player_algorithms[player],
                "co_player_algorithms": [name for index, name in enumerate(self.profile) if index != player],
                **self.final_values[player]}
        common |= {"filename": self.path.name, "algorithm": self.profile[0], "algorithm_label": algorithm_label(self.profile[0]),
                   "feedback_label": FEEDBACK_MODE_LABELS[self.feedback_mode], "environment": info.environment,
                   "environment_label": ENVIRONMENT_LABELS[info.environment], "base_learner_seed": info.base_learner_seed}
        if isinstance(info, ScalingDetails):
            return common | {"run_id": self.run_id, "action_counts": list(info.action_counts), "replicates": info.replicates,
                             "target_regret": info.target_regret, "environment_detail": adversarial_scaling_environment_detail(info._asdict())}
        target = TARGET_REGRET_BY_ALGORITHM.get(self.profile[0], "external")
        return common | {"n_actions": info.n_actions, "base_environment_seed": info.base_environment_seed,
            "environment_seed": info.environment_seed, "learner_seed": info.learner_seed, "replicate": info.replicate,
            "runtime_fingerprint": self.runtime_fingerprint, "target_regret": target,
            "average_regret": self.final_values[0][f"average_{target}_regret"],
            "environment_detail": adversarial_environment_detail(info._asdict()), **self.final_values[0]}


@dataclass(frozen=True)
class ResultSet:
    records: tuple[ResultRecord, ...] = ()
    filenames: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def paths(self) -> list[Path]:
        return [record.path for record in self.records]

    @property
    def replicates(self) -> list[int]:
        return [record.replicate for record in self.records]

    def groups(self, policy: Literal["dashboard", "builder"]) -> list["ResultSet"]:
        """First replicate wins, except ambiguous adversarial builder groups vanish."""
        grouped = defaultdict(list)
        for record in self.records:
            grouped[(record.comparison_key, record.profile)].append(record)
        results = []
        for records in grouped.values():
            by_replicate = {}
            for record in records:
                by_replicate.setdefault(record.replicate, record)
            if policy == "builder" and records[0].kind == "adversarial" and len(by_replicate) != len(records):
                continue
            selected = [by_replicate[key] for key in sorted(by_replicate)]
            results.append(ResultSet(tuple(selected)))
        return results

    def detail_paths(self, group_id: str) -> list[Path]:
        """All matching filenames, including duplicates omitted by summaries."""
        paths = sorted({record.path for record in self.records if record.group_id == group_id}, key=lambda path: path.name)
        if not paths:
            raise KeyError(group_id)
        return paths

    def context_id(self, player: int) -> str:
        first = self.records[0]
        key = (first.kind, first.comparison_key, player, self.replicates)
        return sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]

    def summaries(self, *, grouped: bool = False) -> list[dict]:
        """HTTP/template projection. Arithmetic/order match the historical dashboard."""
        if not grouped:
            return [record.summary(player) for record in self.records
                    for player in range(len(record.profile) if record.kind == "fixed" else 1)]
        summaries = []
        for group in self.groups("dashboard"):
            replicates = group.replicates
            label = str(replicates[0]) if len(replicates) == 1 else (
                f"{replicates[0]}–{replicates[-1]}" if replicates == list(range(replicates[0], replicates[-1] + 1))
                else ", ".join(map(str, replicates)))
            for player in range(len(group.records[0].profile)):
                rows = [record.summary(player) for record in group.records]
                result = dict(rows[0])
                for field in SUMMARY_REGRET_FIELDS:
                    values = [record.final_values[player][field] for record in group.records
                              if field in record.final_values[player]]
                    if len(values) == len(rows):
                        result[field] = float(np.mean(values))
                if group.records[0].kind == "adversarial":
                    result["average_regret"] = result[f"average_{result['target_regret']}_regret"]
                summaries.append(result | {"group_id": group.records[0].group_id, "replicate": replicates[0],
                    "replicates": replicates, "replicate_count": len(replicates), "replicate_label": label, "runs": rows})
        return summaries


class ResultRepository:
    def __init__(self, directory: Path, kind: ResultKind = "fixed"):
        self.directory, self.kind = Path(directory), kind
        self._cache = {}
        self._lock = Lock()

    def _read(self, path: Path) -> tuple[ResultRecord | None, str | None]:
        try:
            return ResultRecord.read(path, self.kind), None
        except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
            return None, f"Skipped {path.name}: {error}"

    def snapshot(self, supported_games=None) -> ResultSet:
        with self._lock:
            records, filenames, warnings, active = [], [], [], {}
            unsupported = {}
            for path in sorted(self.directory.glob("*.csv")):
                # Preserve the fixed index's stat cache and the one-player readers'
                # uncached/error behavior. No new cross-request cache policy.
                if self.kind == "fixed":
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    key = (str(path.absolute()), stat.st_mtime_ns, stat.st_size)
                    record, warning = self._cache[key] if key in self._cache else self._read(path)
                    active[key] = (record, warning)
                else:
                    record, warning = self._read(path)
                filenames.append(path.name)
                if warning:
                    warnings.append(warning)
                if record is not None:
                    if supported_games is not None and record.scope not in supported_games:
                        unsupported[path.name] = record.scope
                    else:
                        records.append(record)
            self._cache = active
            warnings.extend(f"Skipped {name}: unsupported game {game}" for name, game in sorted(unsupported.items()))
            return ResultSet(tuple(records), tuple(filenames), tuple(warnings))
