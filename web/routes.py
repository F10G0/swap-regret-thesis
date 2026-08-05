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
from experiments.scenarios.adversarial import MAX_ADVERSARIAL_ACTIONS
from web.jobs import ServiceBusyError
from web.services import DashboardService, PlotUpdateError
from web.validation import (
    ExperimentForm,
    parse_adversarial_experiment_form,
    parse_experiment_form,
    parse_non_negative_integer,
    parse_positive_integer,
)
from web.view_models import (
    adversarial_context,
    custom_games_context,
    dashboard_context,
    figure_data as build_figure_data,
)


dashboard = Blueprint("dashboard", __name__)


def get_service() -> DashboardService:
    return current_app.extensions["dashboard_service"]


def _dashboard_context(form_state: dict | None = None, inline_error: str | None = None) -> dict:
    return dashboard_context(get_service(), form_state, inline_error)


def _parse_form() -> ExperimentForm:
    service = get_service()
    return parse_experiment_form(
        request.form,
        games=service.game_player_counts,
        algorithms_by_feedback_mode=service.algorithms_by_feedback_mode,
        max_horizon=current_app.config["MAX_HORIZON"],
        max_replicates=current_app.config["MAX_REPLICATES"],
    )


def _submitted_form_state() -> dict:
    service = get_service()
    state = service.default_form_state() | dict(request.form)
    algorithm_names = request.form.getlist("algorithm_names")
    if algorithm_names:
        state["algorithm_names"] = algorithm_names
    return state


def _custom_games_context(form_state: dict | None = None, inline_error: str | None = None, inspection: dict | None = None) -> dict:
    return custom_games_context(get_service(), form_state, inline_error, inspection)


def _adversarial_context(
    form_state: dict | None = None,
    inline_error: str | None = None,
) -> dict:
    return adversarial_context(get_service(), form_state, inline_error)


@dashboard.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", **_dashboard_context())

    try:
        form = _parse_form()
        job = get_service().submit_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return (
            render_template(
                "index.html",
                **_dashboard_context(_submitted_form_state(), str(error)),
            ),
            400,
        )

    flash(f"Queued experiment job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard.route("/adversarial", methods=["GET", "POST"])
def adversarial():
    service = get_service()
    if request.method == "GET":
        return render_template("adversarial.html", **_adversarial_context())

    submitted_state = service.default_adversarial_form_state() | dict(
        request.form
    )
    try:
        form = parse_adversarial_experiment_form(
            request.form,
            algorithms=service.adversarial_algorithms,
            max_actions=MAX_ADVERSARIAL_ACTIONS,
            max_horizon=current_app.config["MAX_HORIZON"],
        )
        job = service.submit_adversarial_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return (
            render_template(
                "adversarial.html",
                **_adversarial_context(submitted_state, str(error)),
            ),
            400,
        )

    flash(f"Queued adversarial job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.adversarial"))


@dashboard.get("/adversarial/experiments/<filename>")
def download_adversarial_experiment(filename: str):
    try:
        filename = get_service().validate_adversarial_csv_filename(filename)
    except (FileNotFoundError, ValueError):
        abort(404)
    return send_from_directory(
        get_service().adversarial_raw_dir.resolve(),
        filename,
        as_attachment=True,
    )


@dashboard.get("/adversarial/figures/<filename>")
def adversarial_figure(filename: str):
    try:
        filename = get_service().validate_adversarial_figure_filename(filename)
    except (FileNotFoundError, ValueError):
        abort(404)
    return send_from_directory(
        get_service().adversarial_figure_dir.resolve(),
        filename,
    )


@dashboard.post("/adversarial/delete-experiment")
def delete_adversarial_experiment():
    try:
        filename = request.form["filename"]
        get_service().delete_adversarial_experiment(filename)
    except (
        KeyError,
        FileNotFoundError,
        PlotUpdateError,
        ServiceBusyError,
        ValueError,
    ) as error:
        flash(str(error), "error")
    else:
        flash(f"Deleted {filename} and rebuilt adversarial figures.", "success")
    return redirect(url_for("dashboard.adversarial"))


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
    return redirect(url_for("dashboard.adversarial"))


@dashboard.route("/custom-games", methods=["GET", "POST"])
def custom_games():
    service = get_service()
    if request.method == "POST":
        try:
            n_players = parse_positive_integer(request.form["n_players"], "number of players", MAX_CUSTOM_PLAYERS)
            action_counts = [
                parse_positive_integer(value, f"player {player} actions", MAX_CUSTOM_ACTIONS_PER_PLAYER)
                for player, value in enumerate(request.form.getlist("action_counts"))
            ]
            seed = parse_non_negative_integer(request.form["seed"], "seed")
            definition = service.create_custom_game(
                request.form["name"],
                n_players,
                action_counts,
                seed,
                request.form.get("payoff_structure", "general_sum"),
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
    json_response = request.accept_mimetypes.best == "application/json"
    try:
        job = get_service().submit_plot_rebuild()
    except ServiceBusyError as error:
        if json_response:
            return jsonify({"error": str(error)}), 409
        flash(str(error), "error")
    else:
        if json_response:
            return jsonify({**job.public_data(), "url": url_for("dashboard.job_status", job_id=job.id)}), 202
        flash(f"Queued plot job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = get_service().jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job.public_data())


@dashboard.get("/figures")
def figure_data():
    return jsonify(build_figure_data(get_service()))


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
    endpoint = (
        "dashboard.adversarial"
        if return_to == "adversarial"
        else "dashboard.index"
    )
    return redirect(url_for(endpoint))


@dashboard.get("/figures/<filename>")
def serve_figure(filename: str):
    try:
        filename = get_service().validate_figure_filename(filename)
    except ValueError:
        abort(404)
    return send_from_directory(
        get_service().figure_dir.resolve(),
        filename,
    )


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
    try:
        filename = get_service().validate_csv_filename(filename)
    except ValueError:
        abort(404)
    return send_from_directory(
        get_service().raw_dir.resolve(),
        filename,
        as_attachment=True,
    )


@dashboard.get("/experiments/<filename>/joint-actions.<figure_format>")
def joint_actions(filename: str, figure_format: str):
    if figure_format not in FIGURE_FORMATS:
        abort(404)
    try:
        path = figure_path(get_service().joint_action_figure(filename), figure_format)
        if not path.is_file():
            raise FileNotFoundError(path)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name)


@dashboard.get("/experiment-groups/<group_id>/joint-actions.<figure_format>")
def group_joint_actions(group_id: str, figure_format: str):
    if figure_format not in FIGURE_FORMATS:
        abort(404)
    try:
        path = figure_path(get_service().group_joint_action_figure(group_id), figure_format)
        if not path.is_file():
            raise FileNotFoundError(path)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name)


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
            "distance",
        ),
        figure_format,
    )


@dashboard.get("/experiment-groups/<group_id>/equilibrium-distance.<figure_format>")
def group_equilibrium_distance(group_id: str, figure_format: str):
    return _equilibrium_convergence_response(
        lambda: get_service().request_group_equilibrium_convergence_figure(
            group_id,
            "distance",
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
