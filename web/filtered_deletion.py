"""Server-authoritative membership for filtered dashboard deletion.

Only presentation filter fields select groups. The existing service still resolves
files for each selected scientific group through ResultSet.detail_paths.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

from experiments.result_catalog import ResultSet
from web.presentation_query import parse_dashboard_query, project_dashboard_query


FILTER_FIELDS = (
    "mode", "scope", "context", "comparison_mode", "comparisonMode",
    "feedback", "horizon", "profiles", "player", "action",
)


@dataclass(frozen=True)
class FilteredDeletionPreview:
    count: int
    membership_digest: str
    group_ids: tuple[str, ...]


class FilteredResultsChanged(ValueError):
    def __init__(self, count: int | None = None):
        self.count = count
        super().__init__("Results changed; please review and confirm again.")


def filtered_deletion_membership(
    results: ResultSet, catalog: Mapping, values: Mapping,
) -> FilteredDeletionPreview:
    """Project all matching groups; ignore sort, display columns and future pages."""
    if not values.get("context"):
        raise ValueError("a result context is required for filtered deletion")
    state = {key: values[key] for key in FILTER_FIELDS if key in values}
    comparison = state.get("comparison_mode", state.get("comparisonMode", "regrets"))
    state["metric"] = "all"
    state["view"] = "horizon_scaling" if comparison == "horizons" else "all"
    query = parse_dashboard_query(state, catalog)
    groups = tuple(sorted(set(project_dashboard_query(results, catalog, query).group_ids)))
    identity = {
        "mode": query.mode, "scope": query.scope, "context": query.context_id,
        "comparison_mode": query.comparison_mode, "feedback": query.feedback,
        "horizon": query.horizon, "profiles": sorted(set(query.profiles)),
        "player": query.player, "action": query.action, "groups": groups,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FilteredDeletionPreview(len(groups), sha256(encoded).hexdigest(), groups)
