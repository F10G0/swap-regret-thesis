from flask import current_app, url_for
from math import sqrt

from config import SEED
from experiments.algorithm_labels import algorithm_profile_label
from experiments.result_schema import REGRET_NAMES
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
)
from experiments.scenarios.cross_play import FEEDBACK_MODE_LABELS
from experiments.game_catalog import (
    CUSTOM_PAYOFF_STRUCTURES,
    MAX_CUSTOM_ACTIONS_PER_PLAYER,
    MAX_CUSTOM_PAYOFF_VALUES,
    MAX_CUSTOM_PLAYERS,
)
from metrics.equilibrium_distance import EquilibriumAnalysisUnavailable
from web.services import DashboardService


REGRET_COLUMNS = [
    {"key": f"{view}_{metric}", "metric": metric, "view": view,
     "label": f"{metric.title()} · {'R/T' if view == 'average' else 'R/√T'}"}
    for metric in REGRET_NAMES for view in ("average", "sqrt_scaling")
]
ADVERSARIAL_ENVIRONMENT_DESCRIPTIONS = {
    HISTORICAL_FREQUENCY_ENVIRONMENT: "The most frequent half of actions (rounded up) receive payoff 0; the rest receive 1.",
    RANDOM_WALK_ENVIRONMENT: "Independent lazy random walks derived reproducibly from the experiment seed.",
}


def _display_regrets(summary):
    # Change display normalization only, using the already aggregated final values.
    return {column["key"]: summary[f"average_{column['metric']}_regret"] *
            (1 if column["view"] == "average" else sqrt(summary["horizon"]))
            for column in REGRET_COLUMNS}


def _recent_jobs(service: DashboardService) -> list[dict]:
    jobs = service.jobs.recent()
    visible = [job for index, job in enumerate(jobs) if index < 5 or job.status in {"queued", "running"}]
    return [
        {**job.public_data(), "url": url_for("dashboard.job_status", job_id=job.id)}
        for job in visible
    ]


def _experiment_page_context(
    service: DashboardService,
    form_state: dict | None,
    inline_error: str | None,
    feedback_modes: dict,
    algorithms_by_feedback_mode: dict,
    default_form_state: dict,
) -> dict:
    return {
        "feedback_modes": feedback_modes,
        "algorithms_by_feedback_mode": algorithms_by_feedback_mode,
        "algorithm_labels": service.algorithm_labels,
        "form_state": form_state or default_form_state,
        "inline_error": inline_error,
        "jobs": _recent_jobs(service),
        "busy": service.jobs.is_busy(),
        "max_horizon": current_app.config["MAX_HORIZON"],
        "max_replicates": current_app.config["MAX_REPLICATES"],
        "regret_columns": REGRET_COLUMNS,
    }


def dashboard_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
    *,
    results=None,
    browse_page=None,
    browsing=None,
) -> dict:
    game_definitions = service.game_definitions
    games = list(game_definitions)
    game_presentations = service.game_presentations
    results = service.result_snapshot() if results is None else results
    summaries = []
    selected_rows = (() if browsing and browsing["bootstrap"] else
                     browse_page.rows if browse_page is not None else results.summaries(grouped=True))
    for projected in selected_rows:
        summary = projected.summary if browse_page is not None else projected
        row = {key: summary[key] for key in (
            "game", "feedback_mode", "seed", "replicate_label", "replicate_count",
            "player", "algorithm_profile", "group_id", "horizon", "stationary_method",
        )}
        row.update({
            "profile_label": algorithm_profile_label(summary["algorithm_profile"]),
            "feedback_label": FEEDBACK_MODE_LABELS[summary["feedback_mode"]],
            "display_regrets": (projected.display_regrets if browse_page is not None
                                else _display_regrets(summary)),
            "best_regret_columns": {
                column["key"] for column in REGRET_COLUMNS
                if browse_page is not None and
                (summary["group_id"], summary["player"], column["key"]) in browse_page.best_cells
            },
        })
        summaries.append(row)

    return {
        **_experiment_page_context(
            service,
            form_state,
            inline_error,
            service.feedback_modes,
            service.algorithms_by_feedback_mode,
            service.default_form_state(),
        ),
        "experiment_mode": "fixed",
        "browsing": browsing,
        "games": games,
        "game_definitions": {game_id: definition.public_data() for game_id, definition in game_definitions.items()},
        "built_in_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "builtin"],
        "custom_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "custom"],
        "game_presentations": game_presentations,
        "summaries": summaries,
        "warnings": list(results.warnings),
    }


def fixed_group_detail_context(service: DashboardService, group_id: str, player: int) -> dict:
    """Project one current dashboard group/player without changing its membership policy."""
    results = service.result_snapshot()
    group = next((group for group in results.groups("dashboard")
                  if group.records[0].group_id == group_id), None)
    if group is None:
        raise KeyError(group_id)
    if not 0 <= player < len(group.records[0].profile):
        raise ValueError("invalid player")
    if any(not path.is_file() for path in group.paths):
        raise KeyError(group_id)
    summary = group.summaries(grouped=True)[player]
    matrix_figures_available = service.supports_matrix_figures(summary["game"])
    equilibrium_distance_available = service.supports_equilibrium_distance(summary["game"])
    equilibrium_distance_unavailable = None
    if equilibrium_distance_available:
        try:
            service.preflight_equilibrium_distance(summary["game"])
        except EquilibriumAnalysisUnavailable as error:
            equilibrium_distance_available = False
            equilibrium_distance_unavailable = str(error)
    return {
        **{key: summary[key] for key in (
            "group_id", "player", "game", "feedback_mode", "horizon", "seed",
            "replicate_label", "replicate_count", "stationary_method", "algorithm_profile",
        )},
        "profile_label": algorithm_profile_label(summary["algorithm_profile"]),
        "feedback_label": FEEDBACK_MODE_LABELS[summary["feedback_mode"]],
        "display_regrets": _display_regrets(summary),
        "runs": [
            {
                "experiment": run["experiment"],
                "replicate": run["replicate"],
                "download_url": url_for("dashboard.download_experiment", filename=run["experiment"]),
            }
            for run in summary["runs"]
        ],
        "joint_actions_url": (
            url_for("dashboard.group_joint_actions", group_id=group_id, figure_format="png")
            if matrix_figures_available else None
        ),
        "joint_actions_pdf_url": (
            url_for("dashboard.group_joint_actions", group_id=group_id, figure_format="pdf")
            if matrix_figures_available else None
        ),
        "equilibrium_distance_unavailable": equilibrium_distance_unavailable,
        "equilibrium_distance_url": (
            url_for("dashboard.group_equilibrium_distance", group_id=group_id, figure_format="png")
            if equilibrium_distance_available else None
        ),
        "equilibrium_distance_pdf_url": (
            url_for("dashboard.group_equilibrium_distance", group_id=group_id, figure_format="pdf")
            if equilibrium_distance_available else None
        ),
    }


def one_player_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
    *,
    results=None,
    browse_page=None,
    browsing=None,
) -> dict:
    results = service.result_snapshot("adversarial") if results is None else results
    if browsing and browsing["bootstrap"]:
        summaries = []
    elif browse_page is not None:
        summaries = [
            dict(projected.summary,
                 display_regrets=projected.display_regrets,
                 best_regret_columns={
                     column["key"] for column in REGRET_COLUMNS
                     if (projected.group_id, 0, column["key"]) in browse_page.best_cells
                 })
            for projected in browse_page.rows
        ]
    else:
        summaries = results.summaries(grouped=True)
        for summary in summaries:
            summary["display_regrets"] = _display_regrets(summary)
            summary["best_regret_columns"] = set()
    return {
        **_experiment_page_context(
            service,
            form_state,
            inline_error,
            FEEDBACK_MODE_LABELS,
            service.adversarial_algorithms_by_feedback_mode,
            service.default_adversarial_form_state(),
        ),
        "experiment_mode": "adversarial",
        "browsing": browsing,
        "adversarial_environments": ENVIRONMENT_LABELS,
        "adversarial_environment_descriptions": ADVERSARIAL_ENVIRONMENT_DESCRIPTIONS,
        "summaries": summaries,
        "warnings": list(results.warnings),
    }


def custom_games_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
    inspection: dict | None = None,
) -> dict:
    definitions, warnings = service.custom_games()
    return {
        "custom_games": definitions,
        "warnings": warnings,
        "inline_error": inline_error,
        "form_state": form_state or {
            "name": "",
            "n_players": 2,
            "seed": SEED,
            "payoff_structure": "zero_sum",
        },
        "inspection": inspection,
        "payoff_structures": CUSTOM_PAYOFF_STRUCTURES,
        "max_players": MAX_CUSTOM_PLAYERS,
        "max_actions": MAX_CUSTOM_ACTIONS_PER_PLAYER,
        "max_payoff_values": MAX_CUSTOM_PAYOFF_VALUES,
    }
