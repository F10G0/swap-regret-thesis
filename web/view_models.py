from flask import current_app, url_for

from config import SEED
from experiments.game_catalog import MAX_CUSTOM_ACTIONS_PER_PLAYER, MAX_CUSTOM_PAYOFF_VALUES, MAX_CUSTOM_PLAYERS
from web.result_groups import aggregate_result_summaries
from web.services import DashboardService
from web.experiment_modes import REGRET_EVALUATIONS
from web.validation import DEFAULT_TRAJECTORY_POINTS, MAX_TRAJECTORY_POINTS, MIN_TRAJECTORY_POINTS


def figure_data(service: DashboardService) -> list[dict]:
    return [
        {**figure, "url": url_for("dashboard.serve_figure", filename=figure["filename"])}
        for figure in service.figure_records()
    ]


def _equilibrium_figure_data(service: DashboardService) -> dict[str, dict[str, str]]:
    return {
        game_name: {
            equilibrium: url_for("dashboard.equilibrium_figure", game_name=game_name, equilibrium=equilibrium)
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
        equilibrium_trajectory_available = service.supports_equilibrium_trajectory(summary["game"])
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
                url_for("dashboard.group_joint_actions", group_id=summary["group_id"])
                if matrix_figures_available
                else None
            ),
            "equilibrium_distance_url": (
                url_for("dashboard.group_equilibrium_distance", group_id=summary["group_id"])
                if equilibrium_distance_available
                else None
            ),
            "equilibrium_trajectory_url": (
                url_for("dashboard.group_equilibrium_trajectory", group_id=summary["group_id"])
                if equilibrium_trajectory_available
                else None
            ),
        })

    jobs = service.jobs.recent()
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
        "default_trajectory_points": DEFAULT_TRAJECTORY_POINTS,
        "min_trajectory_points": MIN_TRAJECTORY_POINTS,
        "max_trajectory_points": MAX_TRAJECTORY_POINTS,
        "players": sorted({summary["player"] for summary in summaries} | {figure["player"] for figure in figures}),
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
        "form_state": form_state or {"name": "", "n_players": 3, "seed": SEED},
        "inspection": inspection,
        "max_players": MAX_CUSTOM_PLAYERS,
        "max_actions": MAX_CUSTOM_ACTIONS_PER_PLAYER,
        "max_payoff_values": MAX_CUSTOM_PAYOFF_VALUES,
    }
