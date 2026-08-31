from flask import current_app, url_for

from config import SEED
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    FEEDBACK_MODE_LABELS,
    INITIALIZATION_LABELS,
    MAX_ADVERSARIAL_ACTIONS,
    RANDOM_WALK_ENVIRONMENT,
)
from experiments.game_catalog import (
    CUSTOM_PAYOFF_STRUCTURES,
    MAX_CUSTOM_ACTIONS_PER_PLAYER,
    MAX_CUSTOM_PAYOFF_VALUES,
    MAX_CUSTOM_PLAYERS,
)
from web.result_groups import aggregate_result_summaries
from web.services import DashboardService
from web.experiment_modes import REGRET_EVALUATION_LABELS


FIGURE_URL_FIELDS = {
    "filename": "url",
    "pdf_filename": "pdf_url",
    "confidence_free_filename": "confidence_free_url",
    "confidence_free_pdf_filename": "confidence_free_pdf_url",
}


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
        "regret_evaluations": REGRET_EVALUATION_LABELS,
        "algorithms_by_feedback_mode": algorithms_by_feedback_mode,
        "algorithm_labels": service.algorithm_labels,
        "form_state": form_state or default_form_state,
        "inline_error": inline_error,
        "jobs": _recent_jobs(service),
        "busy": service.jobs.is_busy(),
        "max_horizon": current_app.config["MAX_HORIZON"],
        "max_replicates": current_app.config["MAX_REPLICATES"],
    }


def _equilibrium_urls(game_name: str) -> dict[str, dict[str, str]]:
    return {
        equilibrium: {
            figure_format: url_for("dashboard.equilibrium_figure", game_name=game_name, equilibrium=equilibrium, figure_format=figure_format)
            for figure_format in ("png", "pdf")
        }
        for equilibrium in ("ce", "cce")
    }


def _equilibrium_figure_data(service: DashboardService) -> dict[str, dict[str, dict[str, str]]]:
    return {
        game_name: _equilibrium_urls(game_name)
        for game_name in service.games
        if service.supports_matrix_figures(game_name)
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
    figures = _figure_data(service.figure_records(), "dashboard.serve_figure")
    summaries = []
    for summary in aggregate_result_summaries(results.summaries):
        profile_label = " vs ".join(service.algorithm_labels.get(name, name) for name in summary["algorithm_profile"])
        matrix_figures_available = service.supports_matrix_figures(summary["game"])
        equilibrium_distance_available = service.supports_equilibrium_distance(summary["game"])
        summaries.append({
            **summary,
            "profile_label": profile_label,
            "regret_sources": [
                source
                for source in ("expected", "realized")
                if any(name.startswith(f"average_{source}_") and name.endswith("_regret") for name in summary)
            ],
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

    for figure in figures:
        game_label = game_presentations[figure["game"]]["label"]
        figure.update(
            scope=figure["game"],
            secondary=str(figure["player"]),
            title=f"{game_label} · {figure['source'].title()} · Player {figure['player']} · {figure['regret'].title()}",
            alt=f"{game_label}, {figure['source']} {figure['regret']} regret for player {figure['player']}",
        )
    players = sorted({summary["player"] for summary in summaries} | {figure["player"] for figure in figures})

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
        "experiments": [
            {
                "filename": filename,
                "download_url": url_for("dashboard.download_experiment", filename=filename),
            }
            for filename in results.filenames
        ],
        "figures": figures,
        "equilibrium_figures": _equilibrium_figure_data(service),
        "summaries": summaries,
        "warnings": results.warnings,
        "players": players,
        "result_filters": [
            {
                "key": "scope",
                "label": "Game",
                "all_label": "All games",
                "options": [(game, game_presentations[game]["label"]) for game in games],
            },
            {
                "key": "secondary",
                "label": "Player",
                "all_label": "All players",
                "options": [(str(player), f"Player {player}") for player in players],
            },
        ],
    }


def one_player_context(
    service: DashboardService,
    form_state: dict | None = None,
    inline_error: str | None = None,
) -> dict:
    summaries, warnings = service.adversarial_result_summaries()
    figures = _figure_data(service.adversarial_figure_records(), "dashboard.adversarial_figure")
    scaling_summaries, scaling_warnings = service.adversarial_scaling_summaries()
    scaling_figures = _figure_data(
        service.adversarial_scaling_figure_records(scaling_summaries),
        "dashboard.adversarial_scaling_figure",
    )
    for summary in summaries:
        summary["regret_sources"] = [source for source in ("expected", "realized") if summary[f"{source}_regret"] is not None]
        summary["download_url"] = url_for("dashboard.download_adversarial_experiment", filename=summary["filename"])
    for summary in scaling_summaries:
        summary["download_url"] = url_for("dashboard.download_adversarial_scaling_experiment", filename=summary["filename"])
    for figure in figures:
        view_label = f"Average {figure['regret'].title()}" if figure["view"] == "average" else f"{figure['regret'].title()} / sqrt(t)"
        figure.update(
            scope=figure["environment"],
            secondary=figure["feedback_mode"],
            title=f"{figure['environment_label']} · {figure['feedback_label']} · {figure['source'].title()} · {view_label} · {figure['n_actions']} actions",
            alt=f"{figure['environment_label']}, {figure['feedback_label']}, {figure['source']} {figure['regret']} regret",
        )
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
        "initialization_modes": INITIALIZATION_LABELS,
        "random_walk_environment": RANDOM_WALK_ENVIRONMENT,
        "summaries": summaries,
        "figures": figures,
        "scaling_summaries": scaling_summaries,
        "scaling_figures": scaling_figures,
        "warnings": warnings + scaling_warnings,
        "max_actions": MAX_ADVERSARIAL_ACTIONS,
        "result_filters": [
            {
                "key": "scope",
                "label": "Environment",
                "all_label": "All environments",
                "options": list(ENVIRONMENT_LABELS.items()),
            },
            {
                "key": "secondary",
                "label": "Feedback",
                "all_label": "All feedback",
                "options": list(FEEDBACK_MODE_LABELS.items()),
            },
        ],
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
        equilibrium_figures = _equilibrium_urls(game_name)
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
