"""Canonical server-side dashboard browsing and complete-group pagination."""

from dataclasses import dataclass, replace
from math import ceil
import re
from urllib.parse import urlencode

from web.presentation_query import (
    DashboardProjection, DashboardQuery, ProjectedGroup, ProjectedRow, REGRET_COLUMNS,
    _compatible_profiles, parse_dashboard_query,
)


PAGE_SIZES = (25, 50, 100)


@dataclass(frozen=True)
class BrowsePage:
    query: DashboardQuery
    page: int
    page_size: int
    total_groups: int
    total_pages: int
    groups: tuple[ProjectedGroup, ...]
    rows: tuple[ProjectedRow, ...]
    best_cells: frozenset[tuple[str, int, str]]


def _positive_decimal(raw, name: str) -> int:
    if not isinstance(raw, str) or re.fullmatch(r"[0-9]+", raw) is None or int(raw) < 1:
        raise ValueError(f"invalid {name}")
    return int(raw)


def _hidden_regret_sort(query: DashboardQuery) -> bool:
    if query.sort not in REGRET_COLUMNS:
        return False
    view, metric = query.sort.rsplit("_", 1)
    return ((query.metric != "all" and query.metric != metric)
            or (query.view not in {"all", "horizon_scaling"} and query.view != view))


def parse_browsing_query(values, catalog, mode: str) -> tuple[DashboardQuery, int, int]:
    """Translate the canonical GET contract through the Phase 3A validator."""
    if "context" not in values or values.get("mode") != mode:
        raise ValueError("incomplete or mismatched dashboard query")
    profiles = values.getlist("profile") if hasattr(values, "getlist") else values.get("profile", ())
    query = parse_dashboard_query({
        "mode": mode, "scope": values.get("scope", ""), "context": values.get("context", ""),
        "comparison_mode": values.get("compare", "regrets"),
        "feedback": values.get("feedback", "full_information"),
        "horizon": values.get("horizon", "all"), "profiles": profiles,
        "metric": values.get("metric", "all"), "view": values.get("view", "all"),
        "player": values.get("player"), "action": values.get("action"),
        "sort": values.get("sort") or None, "direction": values.get("dir", "asc"),
    }, catalog)
    if query.mode == "fixed" and query.player == "all" and query.sort is not None:
        raise ValueError("sorting is unavailable for the all-player view")
    if query.comparison_mode != "profiles" and len(query.profiles) > 1:
        context = next(item for item in catalog["contexts"] if item["id"] == query.context_id)
        compatible = _compatible_profiles(query, context, catalog)
        if (query.comparison_mode not in {"regrets", "horizons"}
                or len(query.profiles) != len(compatible)
                or set(query.profiles) != compatible):
            raise ValueError("multiple profiles are unsupported for this comparison")
    if _hidden_regret_sort(query) or (query.sort is None and query.direction != "asc"):
        query = replace(query, sort=None, direction="asc")
    page = _positive_decimal(values.get("page", "1"), "page")
    page_size = _positive_decimal(values.get("page_size", "25"), "page size")
    if page_size not in PAGE_SIZES:
        raise ValueError("unsupported page size")
    return query, page, page_size


def query_parameters(query: DashboardQuery, page: int = 1, page_size: int = 25) -> list[tuple[str, str]]:
    values = [
        ("mode", query.mode), ("scope", query.scope), ("context", query.context_id),
        ("compare", query.comparison_mode), ("feedback", query.feedback),
        ("horizon", query.horizon),
    ]
    values += [("profile", profile) for profile in sorted(set(query.profiles))]
    values += [("metric", query.metric), ("view", query.view)]
    if query.mode == "fixed" and query.player is not None:
        values.append(("player", str(query.player)))
    if query.mode == "adversarial" and query.action is not None:
        values.append(("action", str(query.action)))
    if query.sort is not None:
        values.append(("sort", query.sort))
        values.append(("dir", query.direction))
    values += [("page", str(page)), ("page_size", str(page_size))]
    return values


def query_string(query: DashboardQuery, page: int = 1, page_size: int = 25) -> str:
    return urlencode(query_parameters(query, page, page_size))


def paginate_projection(projection: DashboardProjection, page: int, page_size: int) -> BrowsePage:
    total = len(projection.groups)
    total_pages = max(1, ceil(total / page_size))
    page = min(page, total_pages)
    groups = projection.groups[(page - 1) * page_size:page * page_size]
    rows = tuple(row for group in groups for row in group.matching_rows)
    return BrowsePage(projection.query, page, page_size, total, total_pages,
                      groups, rows, projection.best_cells)


def default_browsing_query(catalog, mode: str) -> DashboardQuery:
    """Select the first deterministic Builder-compatible settled state."""
    for context in catalog["contexts"]:
        for profile in context["profiles"]:
            for horizon in profile["horizons"]:
                action = (str(profile["actions"][0]) if mode == "adversarial" and profile["actions"]
                          else None)
                for comparison in ("regrets", "profiles"):
                    values = {
                        "mode": mode, "scope": context["scope"], "context": context["id"],
                        "comparison_mode": comparison, "feedback": profile["feedback_mode"],
                        "horizon": str(horizon), "profiles": [profile["id"]],
                        "player": str(context["player"]) if mode == "fixed" else None,
                        "action": action, "metric": "all", "view": "all",
                    }
                    try:
                        return parse_dashboard_query(values, catalog)
                    except ValueError:
                        continue
    return parse_dashboard_query({"mode": mode}, catalog)


def builder_state(query: DashboardQuery) -> dict:
    """Give the existing Builder URL-authoritative initial control values."""
    action = str(query.action) if query.mode == "adversarial" and query.action is not None else ""
    key = f"{query.context_id}:{query.feedback}:{query.comparison_mode}:{action}:{query.horizon}"
    return {
        "scope": query.scope, "context": query.context_id,
        "comparisonMode": query.comparison_mode, "feedback": query.feedback,
        "horizon": query.horizon, "profiles": list(query.profiles),
        "player": str(query.player) if query.mode == "fixed" and query.player is not None else "0",
        "action": action if query.mode == "adversarial" else None,
        "metric": query.metric, "view": query.view,
        "profileMetric": query.metric,
        "selectedAction": action if action != "all" else "",
        "selectedHorizon": query.horizon if query.horizon != "all" else "",
        "selections": {key: list(query.profiles)} if query.context_id else {},
    }
