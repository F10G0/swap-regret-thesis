from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from experiments.game_catalog import MAX_CUSTOM_ACTIONS_PER_PLAYER, MAX_CUSTOM_PLAYERS
from experiments.plots import FIGURE_FORMATS, figure_path
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    INITIALIZATION_LABELS,
    MAX_ADVERSARIAL_ACTIONS,
)
from web.jobs import ServiceBusyError
from web.services import DashboardService, PlotUpdateError
from web.validation import (
    ExperimentForm,
    parse_adversarial_experiment_form,
    parse_adversarial_scaling_form,
    parse_experiment_form,
    parse_non_negative_integer,
    parse_positive_integer,
)
from web.view_models import (
    custom_games_context,
    dashboard_context,
    one_player_context,
)


dashboard = Blueprint("dashboard", __name__)


def get_service() -> DashboardService:
    return current_app.extensions["dashboard_service"]


def _experiment_context(mode: str, form_state: dict | None = None, inline_error: str | None = None) -> dict:
    builder = one_player_context if mode == "adversarial" else dashboard_context
    return builder(get_service(), form_state, inline_error)


def _parse_form() -> ExperimentForm:
    service = get_service()
    return parse_experiment_form(
        request.form,
        games=service.game_player_counts,
        algorithms_by_feedback_mode=service.algorithms_by_feedback_mode,
        max_horizon=current_app.config["MAX_HORIZON"],
        max_replicates=current_app.config["MAX_REPLICATES"],
    )


def _parse_one_player_form(parser, service: DashboardService):
    return parser(
        request.form,
        algorithms_by_feedback_mode=service.adversarial_algorithms_by_feedback_mode,
        environments=set(ENVIRONMENT_LABELS),
        initialization_modes=set(INITIALIZATION_LABELS),
        max_actions=MAX_ADVERSARIAL_ACTIONS,
        max_horizon=current_app.config["MAX_HORIZON"],
        max_replicates=current_app.config["MAX_REPLICATES"],
    )


def _submitted_form_state(default_state: dict) -> dict:
    state = default_state | dict(request.form)
    algorithm_names = request.form.getlist("algorithm_names")
    if algorithm_names:
        state["algorithm_names"] = algorithm_names
    return state


def _custom_games_context(form_state: dict | None = None, inline_error: str | None = None, inspection: dict | None = None) -> dict:
    return custom_games_context(get_service(), form_state, inline_error, inspection)


def _form_error(mode: str, default_state: dict, error: Exception):
    context = _experiment_context(mode, _submitted_form_state(default_state), str(error))
    return render_template("index.html", **context), 400


def _send_result(filename: str, validator, directory, as_attachment: bool = False):
    try:
        filename = validator(filename)
    except (FileNotFoundError, ValueError):
        abort(404)
    return send_from_directory(directory.resolve(), filename, as_attachment=as_attachment)


def _delete_one_player_result(delete_result, message: str):
    try:
        filename = request.form["filename"]
        delete_result(filename)
    except (KeyError, FileNotFoundError, PlotUpdateError, ServiceBusyError, ValueError) as error:
        flash(str(error), "error")
    else:
        flash(message.format(filename=filename), "success")
    return redirect(url_for("dashboard.index", mode="adversarial"))


@dashboard.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        mode = request.args.get("mode", "fixed")
        if mode not in {"fixed", "adversarial"}:
            abort(404)
        return render_template("index.html", **_experiment_context(mode))

    if request.form.get("experiment_type") == "adversarial":
        return _submit_one_player()

    try:
        form = _parse_form()
        job = get_service().submit_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return _form_error("fixed", get_service().default_form_state(), error)

    flash(f"Queued experiment job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index"))


def _submit_one_player():
    service = get_service()
    try:
        form = _parse_one_player_form(parse_adversarial_experiment_form, service)
        job = service.submit_adversarial_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return _form_error("adversarial", service.default_adversarial_form_state(), error)

    flash(f"Queued adversarial job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index", mode="adversarial"))


@dashboard.post("/adversarial/action-scaling")
def adversarial_action_scaling():
    service = get_service()
    try:
        form = _parse_one_player_form(parse_adversarial_scaling_form, service)
        job = service.submit_adversarial_scaling_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return _form_error("adversarial", service.default_adversarial_form_state(), error)

    flash(f"Queued action-space scaling job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index", mode="adversarial"))


@dashboard.get("/adversarial/experiments/<filename>")
def download_adversarial_experiment(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_adversarial_csv_filename, service.adversarial_raw_dir, True)


@dashboard.get("/adversarial/action-scaling/experiments/<filename>")
def download_adversarial_scaling_experiment(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_adversarial_scaling_csv_filename, service.adversarial_scaling_raw_dir, True)


@dashboard.get("/adversarial/figures/<filename>")
def adversarial_figure(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_adversarial_figure_filename, service.adversarial_figure_dir)


@dashboard.get("/adversarial/action-scaling/figures/<filename>")
def adversarial_scaling_figure(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_adversarial_scaling_figure_filename, service.adversarial_scaling_figure_dir)


@dashboard.post("/adversarial/delete-experiment")
def delete_adversarial_experiment():
    return _delete_one_player_result(
        get_service().delete_adversarial_experiment,
        "Deleted {filename} and rebuilt adversarial figures.",
    )


@dashboard.post("/adversarial/action-scaling/delete-experiment")
def delete_adversarial_scaling_experiment():
    return _delete_one_player_result(
        get_service().delete_adversarial_scaling_experiment,
        "Deleted {filename} and rebuilt action-space scaling figures.",
    )


@dashboard.post("/adversarial/results/clear")
def clear_adversarial_results():
    try:
        csv_count, figure_count = get_service().clear_adversarial_results()
    except ServiceBusyError as error:
        flash(str(error), "error")
    else:
        flash(
            f"Deleted {csv_count} adversarial result(s) and "
            f"{figure_count} figure(s).",
            "success",
        )
    return redirect(url_for("dashboard.index", mode="adversarial"))


@dashboard.route("/custom-games", methods=["GET", "POST"])
def custom_games():
    service = get_service()
    if request.method == "POST":
        try:
            n_players = parse_positive_integer(request.form["n_players"], "number of players", MAX_CUSTOM_PLAYERS)
            payoff_structure = request.form.get("payoff_structure", "general_sum")
            action_counts = [
                parse_positive_integer(value, f"player {player} actions", MAX_CUSTOM_ACTIONS_PER_PLAYER)
                for player, value in enumerate(request.form.getlist("action_counts"))
            ]
            if payoff_structure == "zero_sum" and len(action_counts) == 1:
                action_counts *= 2
            seed = parse_non_negative_integer(request.form["seed"], "seed")
            definition = service.create_custom_game(
                request.form["name"],
                n_players,
                action_counts,
                seed,
                payoff_structure,
            )
        except (FileExistsError, KeyError, OSError, ValueError) as error:
            form_state = dict(request.form)
            form_state["action_counts"] = request.form.getlist("action_counts")
            return render_template("custom_games.html", **_custom_games_context(form_state, str(error))), 400
        flash(f"Created custom game {definition.label}.", "success")
        return redirect(url_for("dashboard.custom_games"))

    return render_template("custom_games.html", **_custom_games_context())


@dashboard.get("/custom-games/<game_id>")
def inspect_custom_game(game_id: str):
    try:
        inspection = get_service().custom_game_inspection(game_id)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return render_template("custom_games.html", **_custom_games_context(inspection=inspection))


@dashboard.get("/custom-games/<game_id>/payoff-slice")
def custom_game_payoff_slice(game_id: str):
    try:
        payoff_player = int(request.args.get("payoff_player", ""))
        row_player = int(request.args.get("row_player", ""))
        column_player = int(request.args.get("column_player", ""))
        fixed_actions = [int(value) for value in request.args.getlist("fixed_action")]
    except ValueError as error:
        return jsonify({"error": f"invalid payoff-slice parameters: {error}"}), 400
    try:
        data = get_service().custom_game_payoff_slice(game_id, payoff_player, row_player, column_player, fixed_actions)
    except (FileNotFoundError, KeyError):
        abort(404)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(data)


@dashboard.get("/custom-games/<game_id>/download")
def download_custom_game(game_id: str):
    try:
        path = get_service().custom_game_file(game_id)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name, as_attachment=True)


@dashboard.post("/custom-games/delete")
def delete_custom_game():
    try:
        definition = get_service().delete_custom_game(request.form["game_id"])
    except (FileNotFoundError, KeyError, OSError, ServiceBusyError, ValueError) as error:
        flash(str(error), "error")
    else:
        flash(f"Deleted custom game {definition.label}.", "success")
    return redirect(url_for("dashboard.custom_games"))


@dashboard.post("/plots/rebuild")
def rebuild_plots():
    try:
        job = get_service().submit_plot_rebuild()
    except ServiceBusyError as error:
        flash(str(error), "error")
    else:
        flash(f"Queued plot job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = get_service().jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job.public_data())


@dashboard.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    try:
        get_service().jobs.cancel(job_id)
    except KeyError:
        abort(404)
    except ValueError as error:
        flash(str(error), "error")
    else:
        flash("Cancellation requested.", "success")
    return_to = request.form.get("return_to")
    return redirect(
        url_for(
            "dashboard.index",
            mode="adversarial" if return_to == "adversarial" else "fixed",
        )
    )


@dashboard.get("/figures/<filename>")
def serve_figure(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_figure_filename, service.figure_dir)


@dashboard.get("/games/<game_name>/equilibria/<equilibrium>.<figure_format>")
def equilibrium_figure(game_name: str, equilibrium: str, figure_format: str):
    try:
        path = get_service().equilibrium_figure(game_name, equilibrium, figure_format)
    except (KeyError, ValueError):
        abort(404)
    except FileNotFoundError as error:
        return jsonify({"status": "failed", "error": str(error)}), 500
    return send_from_directory(path.parent.resolve(), path.name)


@dashboard.get("/experiments/<filename>")
def download_experiment(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_csv_filename, service.raw_dir, True)


@dashboard.get("/experiments/<filename>/joint-actions.<figure_format>")
def joint_actions(filename: str, figure_format: str):
    return _generated_figure_response(
        lambda: get_service().joint_action_figure(filename),
        figure_format,
    )


def _generated_figure_response(generate_figure, figure_format: str):
    if figure_format not in FIGURE_FORMATS:
        abort(404)
    try:
        path = figure_path(generate_figure(), figure_format)
        if not path.is_file():
            raise FileNotFoundError(path)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name)


@dashboard.get("/experiment-groups/<group_id>/joint-actions.<figure_format>")
def group_joint_actions(group_id: str, figure_format: str):
    return _generated_figure_response(
        lambda: get_service().group_joint_action_figure(group_id),
        figure_format,
    )


def _equilibrium_convergence_response(request_figure, figure_format: str):
    if figure_format not in FIGURE_FORMATS:
        abort(404)
    try:
        path, error = request_figure()
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    if error is not None:
        return jsonify({"status": "failed", "error": error}), 500
    if path is None:
        response = jsonify({"status": "generating", "message": "Computing equilibrium convergence…"})
        response.status_code = 202
        response.headers["Retry-After"] = "2"
        return response
    requested_path = figure_path(path, figure_format)
    if not requested_path.is_file():
        abort(404)
    return send_from_directory(requested_path.parent.resolve(), requested_path.name)


@dashboard.get("/experiments/<filename>/equilibrium-distance.<figure_format>")
def equilibrium_distance(filename: str, figure_format: str):
    return _equilibrium_convergence_response(
        lambda: get_service().request_equilibrium_convergence_figure(
            filename,
        ),
        figure_format,
    )


@dashboard.get("/experiment-groups/<group_id>/equilibrium-distance.<figure_format>")
def group_equilibrium_distance(group_id: str, figure_format: str):
    return _equilibrium_convergence_response(
        lambda: get_service().request_group_equilibrium_convergence_figure(
            group_id,
        ),
        figure_format,
    )


@dashboard.post("/delete-experiment")
def delete_experiment():
    try:
        filename = request.form["filename"]
        get_service().delete_experiment(filename)
    except (
        KeyError,
        FileNotFoundError,
        PlotUpdateError,
        ServiceBusyError,
        ValueError,
    ) as error:
        flash(str(error), "error")
    else:
        flash(f"Deleted {filename} and rebuilt the figures.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard.post("/reset")
def reset_results():
    if request.form.get("confirmation") != "reset-results":
        flash("Reset confirmation was missing.", "error")
        return redirect(url_for("dashboard.index"))

    try:
        get_service().clear_results()
    except ServiceBusyError as error:
        flash(str(error), "error")
    else:
        flash("Deleted generated CSV and figure files.", "success")
    return redirect(url_for("dashboard.index"))
