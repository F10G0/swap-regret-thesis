from flask import current_app, url_for

from config import SEED
from experiments.scenarios.adversarial import MAX_ADVERSARIAL_ACTIONS
from experiments.game_catalog import (
    CUSTOM_PAYOFF_STRUCTURES,
    MAX_CUSTOM_ACTIONS_PER_PLAYER,
    MAX_CUSTOM_PAYOFF_VALUES,
    MAX_CUSTOM_PLAYERS,
)
from web.result_groups import aggregate_result_summaries
from web.services import DashboardService
from web.experiment_modes import REGRET_EVALUATIONS


def figure_data(service: DashboardService) -> list[dict]:
    figures = []
    for figure in service.figure_records():
        figures.append({
            **figure,
            "url": url_for("dashboard.serve_figure", filename=figure["filename"]),
            "pdf_url": (
                url_for("dashboard.serve_figure", filename=figure["pdf_filename"])
                if figure["pdf_filename"]
                else None
            ),
        })
    return figures


def _equilibrium_figure_data(service: DashboardService) -> dict[str, dict[str, dict[str, str]]]:
    return {
        game_name: {
            equilibrium: {
                "png": url_for("dashboard.equilibrium_figure", game_name=game_name, equilibrium=equilibrium, figure_format="png"),
                "pdf": url_for("dashboard.equilibrium_figure", game_name=game_name, equilibrium=equilibrium, figure_format="pdf"),
            }
            for equilibrium in ("ce", "cce")
        }
        for game_name in service.games
        if service.supports_matrix_figures(game_name)
    }


def dashboard_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
) -> dict:
    game_definitions = service.game_definitions
    results = service.result_snapshot()
    figures = figure_data(service)
    summaries = []
    for summary in aggregate_result_summaries(results.summaries):
        profile_label = " vs ".join(service.algorithm_labels.get(name, name) for name in summary["algorithm_profile"])
        matrix_figures_available = service.supports_matrix_figures(summary["game"])
        equilibrium_distance_available = service.supports_equilibrium_distance(summary["game"])
        summaries.append({
            **summary,
            "profile_label": profile_label,
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

    jobs = service.jobs.recent()[:5]
    return {
        "games": service.games,
        "game_definitions": {game_id: definition.public_data() for game_id, definition in game_definitions.items()},
        "built_in_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "builtin"],
        "custom_games": [game_id for game_id, definition in game_definitions.items() if definition.source == "custom"],
        "game_presentations": service.game_presentations,
        "feedback_modes": service.feedback_modes,
        "regret_evaluations": REGRET_EVALUATIONS,
        "algorithms_by_feedback_mode": service.algorithms_by_feedback_mode,
        "algorithm_labels": service.algorithm_labels,
        "experiments": results.filenames,
        "figures": figures,
        "equilibrium_figures": _equilibrium_figure_data(service),
        "summaries": summaries,
        "warnings": results.warnings,
        "jobs": [{**job.public_data(), "url": url_for("dashboard.job_status", job_id=job.id)} for job in jobs],
        "busy": service.jobs.is_busy(),
        "form_state": form_state or service.default_form_state(),
        "inline_error": inline_error,
        "max_horizon": current_app.config["MAX_HORIZON"],
        "max_replicates": current_app.config["MAX_REPLICATES"],
        "players": sorted({summary["player"] for summary in summaries} | {figure["player"] for figure in figures}),
    }


def adversarial_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
) -> dict:
    summaries, warnings = service.adversarial_result_summaries()
    figures = [
        {
            **record,
            "url": url_for(
                "dashboard.adversarial_figure",
                filename=record["filename"],
            ),
            "pdf_url": (
                url_for("dashboard.adversarial_figure", filename=record["pdf_filename"])
                if record["pdf_filename"]
                else None
            ),
        }
        for record in service.adversarial_figure_records()
    ]
    return {
        "algorithms": service.adversarial_algorithms,
        "algorithm_labels": service.algorithm_labels,
        "form_state": form_state or service.default_adversarial_form_state(),
        "inline_error": inline_error,
        "summaries": [
            {
                **summary,
                "download_url": url_for(
                    "dashboard.download_adversarial_experiment",
                    filename=summary["filename"],
                ),
            }
            for summary in summaries
        ],
        "figures": figures,
        "warnings": warnings,
        "jobs": [
            {
                **job.public_data(),
                "url": url_for("dashboard.job_status", job_id=job.id),
            }
            for job in service.jobs.recent()[:5]
        ],
        "busy": service.jobs.is_busy(),
        "max_actions": MAX_ADVERSARIAL_ACTIONS,
        "max_horizon": current_app.config["MAX_HORIZON"],
    }


def custom_games_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
    inspection: dict | None = None,
) -> dict:
    definitions, warnings = service.custom_games()
    equilibrium_figures = None
    if inspection and service.supports_matrix_figures(inspection["definition"]["id"]):
        game_name = inspection["definition"]["id"]
        equilibrium_figures = {
            equilibrium: {
                figure_format: url_for(
                    "dashboard.equilibrium_figure",
                    game_name=game_name,
                    equilibrium=equilibrium,
                    figure_format=figure_format,
                )
                for figure_format in ("png", "pdf")
            }
            for equilibrium in ("ce", "cce")
        }
    return {
        "custom_games": definitions,
        "warnings": warnings,
        "inline_error": inline_error,
        "form_state": form_state or {
            "name": "",
            "n_players": 3,
            "seed": SEED,
            "payoff_structure": "general_sum",
        },
        "inspection": inspection,
        "equilibrium_figures": equilibrium_figures,
        "payoff_structures": CUSTOM_PAYOFF_STRUCTURES,
        "max_players": MAX_CUSTOM_PLAYERS,
        "max_actions": MAX_CUSTOM_ACTIONS_PER_PLAYER,
        "max_payoff_values": MAX_CUSTOM_PAYOFF_VALUES,
    }
