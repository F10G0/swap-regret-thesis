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
from web.services import DashboardService


FIGURE_URL_FIELDS = {
    "filename": "url",
    "pdf_filename": "pdf_url",
}

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


def _figure_data(records: list[dict], endpoint: str) -> list[dict]:
    figures = []
    for figure in records:
        data = dict(figure)
        for filename_field, url_field in FIGURE_URL_FIELDS.items():
            if filename_field in figure:
                filename = figure[filename_field]
                data[url_field] = url_for(endpoint, filename=filename) if filename else None
        figures.append(data)
    return figures


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
) -> dict:
    game_definitions = service.game_definitions
    games = list(game_definitions)
    game_presentations = service.game_presentations
    results = service.result_snapshot()
    summaries = []
    for summary in results.summaries(grouped=True):
        matrix_figures_available = service.supports_matrix_figures(summary["game"])
        equilibrium_distance_available = service.supports_equilibrium_distance(summary["game"])
        summaries.append({
            **summary,
            "profile_label": algorithm_profile_label(summary["algorithm_profile"]),
            "display_regrets": _display_regrets(summary),
            "runs": [
                {
                    **run,
                    "download_url": url_for("dashboard.download_experiment", filename=run["experiment"]),
                }
                for run in summary["runs"]
            ],
            "joint_actions_url": (
                url_for("dashboard.group_joint_actions", group_id=summary["group_id"], figure_format="png")
                if matrix_figures_available
                else None
            ),
            "joint_actions_pdf_url": (
                url_for("dashboard.group_joint_actions", group_id=summary["group_id"], figure_format="pdf")
                if matrix_figures_available
                else None
            ),
            "equilibrium_distance_url": (
                url_for("dashboard.group_equilibrium_distance", group_id=summary["group_id"], figure_format="png")
                if equilibrium_distance_available
                else None
            ),
            "equilibrium_distance_pdf_url": (
                url_for("dashboard.group_equilibrium_distance", group_id=summary["group_id"], figure_format="pdf")
                if equilibrium_distance_available
                else None
            ),
        })

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
        "games": games,
        "game_definitions": {game_id: definition.public_data() for game_id, definition in game_definitions.items()},
        "built_in_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "builtin"],
        "custom_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "custom"],
        "game_presentations": game_presentations,
        "summaries": summaries,
        "warnings": list(results.warnings),
    }


def one_player_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
) -> dict:
    results = service.result_snapshot("adversarial")
    scaling = service.result_snapshot("scaling")
    summaries = results.summaries(grouped=True)
    scaling_figures = _figure_data(
        service.adversarial_scaling_figure_records(scaling),
        "dashboard.adversarial_scaling_figure",
    )
    for summary in summaries:
        summary["display_regrets"] = _display_regrets(summary)
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
        "adversarial_environments": ENVIRONMENT_LABELS,
        "adversarial_environment_descriptions": ADVERSARIAL_ENVIRONMENT_DESCRIPTIONS,
        "summaries": summaries,
        "scaling_figures": scaling_figures,
        "warnings": list(results.warnings + scaling.warnings),
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
