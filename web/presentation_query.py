"""Read-only presentation queries over complete scientific dashboard groups.

This module drives dashboard browsing without altering ResultSet membership.
A fixed group stays whole even when only one player row matches.
"""

from dataclasses import dataclass
from functools import cmp_to_key
from math import sqrt
import re
from typing import Mapping

from experiments.algorithm_labels import algorithm_profile_label
from experiments.result_catalog import ResultSet
from experiments.result_schema import REGRET_NAMES
from experiments.scenarios.cross_play import FEEDBACK_MODE_LABELS


REGRET_COLUMNS = tuple(f"{view}_{metric}" for metric in REGRET_NAMES
                       for view in ("average", "sqrt_scaling"))
FIXED_SORTS = ("game", "feedback", "seed_replicates", "player", "profile", "horizon",
               *REGRET_COLUMNS, "actions")
ADVERSARIAL_SORTS = ("environment", "feedback", "algorithm", "action",
                     "configuration", "seed_replicates", "horizon", *REGRET_COLUMNS, "actions")
COMPARISON_MODES = {"fixed": {"regrets", "profiles", "horizons"},
                    "adversarial": {"regrets", "profiles", "actions", "horizons"}}


class DashboardSelectionUnavailable(ValueError):
    """A well-formed browsing selection is absent from the current catalog."""


@dataclass(frozen=True)
class DashboardQuery:
    mode: str = "fixed"
    scope: str = ""
    context_id: str = ""
    comparison_mode: str = "regrets"
    feedback: str = "full_information"
    horizon: str = "all"
    profiles: tuple[str, ...] = ()
    metric: str = "all"
    view: str = "all"
    player: int | str | None = None
    action: int | str | None = None
    sort: str | None = None
    direction: str = "asc"

    @property
    def pageable(self) -> bool:
        # Explicit all-player sorting is a legacy row-level view, not a group order.
        return not (self.mode == "fixed" and self.player == "all")


@dataclass(frozen=True)
class ProjectedRow:
    group_id: str
    player: int
    summary: dict
    display_regrets: dict[str, float]


@dataclass(frozen=True)
class ProjectedGroup:
    group: ResultSet
    canonical_rank: int
    matching_rows: tuple[ProjectedRow, ...]

    @property
    def group_id(self) -> str:
        return self.group.records[0].group_id


@dataclass(frozen=True)
class DashboardProjection:
    query: DashboardQuery
    groups: tuple[ProjectedGroup, ...]
    rows: tuple[ProjectedRow, ...]
    global_minima: dict[tuple[str, ...], float]
    best_cells: frozenset[tuple[str, int, str]]
    pageable: bool
    sort_applied: bool

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups)


def _decimal(value, name: str, minimum: int) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"invalid {name}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"invalid {name}")
    return number


def _context(query: DashboardQuery, catalog: Mapping) -> Mapping | None:
    if not query.context_id:
        return None
    context = next((item for item in catalog.get("contexts", ())
                    if item["id"] == query.context_id), None)
    if context is None:
        raise DashboardSelectionUnavailable("result context is unavailable")
    if context["mode"] != query.mode or context["scope"] != query.scope:
        raise ValueError("result context does not match the selected scope")
    return context


def _compatible_profiles(query: DashboardQuery, context: Mapping, catalog: Mapping) -> set[str]:
    metric_ids = {item["id"] for item in catalog.get("metrics", ())}
    compatible = set()
    for profile in context["profiles"]:
        if query.feedback != "both" and profile["feedback_mode"] != query.feedback:
            continue
        if query.comparison_mode in {"regrets", "horizons"} and not metric_ids <= set(profile["metrics"]):
            continue
        if query.mode == "fixed":
            horizons = profile["horizons"]
            available = len(horizons) >= 2 if query.horizon == "all" else int(query.horizon) in horizons
        elif query.comparison_mode == "actions":
            available = any(int(query.horizon) in horizons
                            for horizons in profile["availability"].values())
        else:
            horizons = profile["availability"].get(str(query.action), ())
            available = len(horizons) >= 2 if query.horizon == "all" else int(query.horizon) in horizons
        if available:
            compatible.add(profile["id"])
    return compatible


def validate_dashboard_query(query: DashboardQuery, catalog: Mapping) -> Mapping | None:
    if query.mode not in COMPARISON_MODES or query.comparison_mode not in COMPARISON_MODES[query.mode]:
        raise ValueError("invalid dashboard mode or comparison mode")
    if query.feedback not in {*FEEDBACK_MODE_LABELS, "both"}:
        raise ValueError("invalid feedback")
    if query.horizon != "all":
        _decimal(query.horizon, "horizon", 1)
    if query.metric not in {"all", *(item["id"] for item in catalog.get("metrics", ()))}:
        raise ValueError("invalid regret metric")
    if query.view not in {"all", *(item["id"] for item in catalog.get("views", ()))}:
        raise ValueError("invalid regret view")
    if query.context_id:
        if query.comparison_mode == "horizons":
            if query.horizon != "all" or query.metric != "all" or query.view != "horizon_scaling":
                raise ValueError("invalid horizon comparison state")
        elif query.comparison_mode == "regrets":
            if query.horizon == "all" or query.metric != "all" or query.view == "horizon_scaling":
                raise ValueError("invalid regret comparison state")
        elif query.horizon == "all" or query.view == "horizon_scaling":
            raise ValueError("invalid comparison state")
    if query.mode == "fixed":
        if query.action is not None or (query.player not in (None, "all")
                                        and (not isinstance(query.player, int) or query.player < 0)):
            raise ValueError("invalid fixed player")
    elif query.player is not None or (query.action not in (None, "all")
                                      and (not isinstance(query.action, int) or query.action < 2)):
        raise ValueError("invalid one-player action")
    if query.mode == "adversarial" and (query.action == "all") != (query.comparison_mode == "actions"):
        raise ValueError("invalid action comparison state")
    if query.direction not in {"asc", "desc"} or (query.sort is not None
            and query.sort not in (FIXED_SORTS if query.mode == "fixed" else ADVERSARIAL_SORTS)):
        raise ValueError("invalid dashboard sort")
    if not isinstance(query.profiles, tuple) or any(not isinstance(p, str) or not p for p in query.profiles):
        raise ValueError("invalid profiles")
    context = _context(query, catalog)
    if context is None:
        if query.profiles:
            raise ValueError("profiles require a result context")
        return None
    if query.comparison_mode not in context["comparison_modes"]:
        raise DashboardSelectionUnavailable("comparison mode unavailable in result context")
    if query.feedback != "both" and query.feedback not in context["feedback_modes"]:
        raise DashboardSelectionUnavailable("feedback unavailable in result context")
    if query.mode == "fixed" and query.player not in (None, "all", context["player"]):
        raise ValueError("player does not match result context")
    if query.horizon != "all" and int(query.horizon) not in context["horizons"]:
        raise DashboardSelectionUnavailable("horizon is unavailable in result context")
    if (query.mode == "adversarial" and query.action not in (None, "all")
            and query.action not in context["actions"]):
        raise DashboardSelectionUnavailable("action is unavailable in result context")
    if not set(query.profiles) <= _compatible_profiles(query, context, catalog):
        raise DashboardSelectionUnavailable("profile is unavailable for this result context")
    return context


def parse_dashboard_query(values: Mapping, catalog: Mapping) -> DashboardQuery:
    """Parse the current Builder selection; missing context means no selected rows."""
    mode = values.get("mode", "fixed")
    raw_player = values.get("player")
    if mode == "adversarial" and raw_player == "0":
        # The current one-player Builder carries this unused fixed-player field.
        raw_player = None
    raw_action = values.get("action")
    if ("comparison_mode" in values and "comparisonMode" in values
            and values["comparison_mode"] != values["comparisonMode"]):
        raise ValueError("conflicting comparison modes")
    comparison_mode = values.get("comparison_mode", values.get("comparisonMode", "regrets"))
    raw_profiles = values.get("profiles", ())
    if isinstance(raw_profiles, str):
        profiles = (raw_profiles,) if raw_profiles else ()
    elif isinstance(raw_profiles, (tuple, list)):
        profiles = tuple(raw_profiles)
    else:
        raise ValueError("invalid profiles")
    query = DashboardQuery(
        mode=mode,
        scope=values.get("scope", ""),
        context_id=values.get("context", ""),
        comparison_mode=comparison_mode,
        feedback=values.get("feedback", "full_information"),
        horizon=str(values.get("horizon", "all")),
        profiles=profiles,
        metric=values.get("metric", "all"),
        view=values.get("view", "all"),
        player=("all" if raw_player == "all" else
                _decimal(str(raw_player), "player", 0) if raw_player is not None else None),
        action=("all" if raw_action == "all" else
                _decimal(str(raw_action), "action", 2) if raw_action is not None else None),
        sort=values.get("sort"),
        direction=values.get("direction", "asc"),
    )
    validate_dashboard_query(query, catalog)
    return query


def _display_regrets(summary: Mapping) -> dict[str, float]:
    return {f"average_{metric}": summary[f"average_{metric}_regret"]
            for metric in REGRET_NAMES} | {
        f"sqrt_scaling_{metric}": summary[f"average_{metric}_regret"] * sqrt(summary["horizon"])
        for metric in REGRET_NAMES
    }


def _matches(summary: Mapping, query: DashboardQuery) -> bool:
    scope = summary["game"] if query.mode == "fixed" else summary["environment"]
    profile = ("_vs_".join(summary["algorithm_profile"]) if query.mode == "fixed"
               else summary["algorithm"])
    if scope != query.scope or (query.feedback != "both" and summary["feedback_mode"] != query.feedback):
        return False
    if query.horizon != "all" and summary["horizon"] != int(query.horizon):
        return False
    if profile not in query.profiles:
        return False
    if query.mode == "fixed":
        return query.player == "all" or summary["player"] == query.player
    return query.action == "all" or summary["n_actions"] == query.action


def _sort_value(row: ProjectedRow, query: DashboardQuery, presentations: Mapping) -> int | float | str:
    summary = row.summary
    column = query.sort
    if column in REGRET_COLUMNS:
        return row.display_regrets[column]
    if column == "horizon":
        return summary["horizon"]
    if column == "feedback":
        return FEEDBACK_MODE_LABELS[summary["feedback_mode"]].casefold()
    if column == "seed_replicates":
        seed = summary["seed"] if query.mode == "fixed" else summary["base_learner_seed"]
        label = f'{seed} / {summary["replicate_label"]}'
        if summary["replicate_count"] > 1:
            label += f' (n={summary["replicate_count"]})'
        return label.casefold()
    if column == "actions":
        return "delete experiment"
    if query.mode == "fixed":
        if column == "game":
            item = presentations.get(summary["game"], {})
            return item.get("label", summary["game"]).casefold()
        if column == "player":
            return summary["player"]
        return algorithm_profile_label(summary["algorithm_profile"]).casefold()
    if column == "environment":
        return summary["environment_label"].casefold()
    if column == "algorithm":
        return summary["algorithm_label"].casefold()
    if column == "action":
        return summary["n_actions"]
    return summary["environment_detail"].casefold()


def _highlight_bucket(row: ProjectedRow, column: str, mode: str) -> tuple[str, ...]:
    summary = row.summary
    return (column,
            summary["game"] if mode == "fixed" else summary["environment"],
            str(row.player), summary["feedback_mode"], str(summary["horizon"]),
            str(summary["seed"] if mode == "fixed" else summary["base_learner_seed"]),
            summary["stationary_method"] if mode == "fixed" else "",
            "", summary["environment_detail"] if mode == "adversarial" else "")


def project_dashboard_query(results: ResultSet, catalog: Mapping, query: DashboardQuery,
                            *, presentations: Mapping | None = None) -> DashboardProjection:
    """Filter the full snapshot; retain complete groups and never page-slice."""
    context = validate_dashboard_query(query, catalog)
    groups = []
    allowed = set(context["result_keys"]) if context is not None else set()
    for rank, group in enumerate(results.groups("dashboard")):
        if group.records[0].group_id not in allowed:
            continue
        rows = tuple(ProjectedRow(summary["group_id"],
                                  summary["player"] if query.mode == "fixed" else 0, summary,
                                  _display_regrets(summary))
                     for summary in group.summaries(grouped=True) if _matches(summary, query))
        if rows:
            groups.append(ProjectedGroup(group, rank, rows))
    sort_applied = query.sort is not None and query.pageable
    if sort_applied:
        def compare(left: ProjectedGroup, right: ProjectedGroup) -> int:
            left_value = _sort_value(left.matching_rows[0], query, presentations or {})
            right_value = _sort_value(right.matching_rows[0], query, presentations or {})
            difference = (left_value > right_value) - (left_value < right_value)
            return (difference if query.direction == "asc" else -difference) or (
                left.canonical_rank - right.canonical_rank)
        groups.sort(key=cmp_to_key(compare))
    rows = tuple(row for group in groups for row in group.matching_rows)
    minima = {}
    for row in rows:
        for column, value in row.display_regrets.items():
            view, metric = column.rsplit("_", 1)
            if query.metric != "all" and metric != query.metric:
                continue
            if query.view not in {"all", "horizon_scaling"} and view != query.view:
                continue
            bucket = _highlight_bucket(row, column, query.mode)
            minima[bucket] = min(value, minima.get(bucket, value))
    best = frozenset((row.group_id, row.player, column)
                     for row in rows for column, value in row.display_regrets.items()
                     if _highlight_bucket(row, column, query.mode) in minima
                     and value == minima[_highlight_bucket(row, column, query.mode)])
    return DashboardProjection(query, tuple(groups), rows, minima, best, query.pageable, sort_applied)
