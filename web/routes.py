import re

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from pypdf.errors import PyPdfError

from experiments.game_catalog import MAX_CUSTOM_ACTIONS_PER_PLAYER, MAX_CUSTOM_PLAYERS
from experiments.plots import FIGURE_FORMATS, figure_path
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    MAX_ADVERSARIAL_ACTIONS,
)
from metrics.equilibrium_distance import EquilibriumAnalysisUnavailable
from web.browsing import (
    builder_state, default_browsing_query, paginate_projection,
    parse_browsing_query, query_parameters, query_string,
)
from web.filtered_deletion import FilteredResultsChanged
from web.jobs import ServiceBusyError
from web.presentation_query import project_dashboard_query
from web.pdf_export import merged_figure_pdf
from web.services import DashboardService
from web.validation import (
    ExperimentForm,
    parse_adversarial_experiment_form,
    parse_experiment_form,
    parse_profile_selection,
    parse_non_negative_integer,
    parse_positive_integer,
)
from web.view_models import (
    custom_games_context,
    dashboard_context,
    fixed_group_detail_context,
    one_player_context,
)


dashboard = Blueprint("dashboard", __name__)


def get_service() -> DashboardService:
    return current_app.extensions["dashboard_service"]


def _browse_url(query, page: int = 1, page_size: int = 25) -> str:
    return f"{url_for('dashboard.index')}?{query_string(query, page, page_size)}"


def _browsing_navigation(page=None, *, bootstrap_query=None) -> dict:
    if page is None:
        return {
            "bootstrap": True, "state": None, "default_url": _browse_url(bootstrap_query),
            "page": 1, "page_size": 25, "total_pages": 1, "total_groups": 0,
        }
    query = page.query
    return {
        "bootstrap": False, "state": builder_state(query), "default_url": "",
        "page": page.page, "page_size": page.page_size,
        "total_pages": page.total_pages, "total_groups": page.total_groups,
        "previous_url": _browse_url(query, page.page - 1, page.page_size) if page.page > 1 else None,
        "next_url": _browse_url(query, page.page + 1, page.page_size)
            if page.page < page.total_pages else None,
        "form_params": [(key, value) for key, value in query_parameters(query)
                        if key not in {"page", "page_size"}],
    }


def _experiment_context(mode: str, form_state: dict | None = None, inline_error: str | None = None,
                        *, results=None, browse_page=None, browsing=None) -> dict:
    service = get_service()
    builder = one_player_context if mode == "adversarial" else dashboard_context
    if browsing is None:
        results = service.result_snapshot(mode) if results is None else results
        catalog = service.figure_builder.catalog(mode, results)
        query = default_browsing_query(catalog, mode)
        projection = project_dashboard_query(results, catalog, query,
                                             presentations=service.game_presentations if mode == "fixed" else {})
        browse_page = paginate_projection(projection, 1, 25)
        browsing = _browsing_navigation(browse_page)
    return builder(service, form_state, inline_error, results=results,
                   browse_page=browse_page, browsing=browsing)


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
    if request.accept_mimetypes.best == "application/json":
        return jsonify(error=str(error)), 400
    context = _experiment_context(mode, _submitted_form_state(default_state), str(error))
    return render_template("index.html", **context), 400


def _queued_experiment_response(job, mode: str, message: str):
    if request.accept_mimetypes.best == "application/json":
        data = job.public_data() | {"url": url_for("dashboard.job_status", job_id=job.id)}
        return jsonify(
            job=data,
            job_html=render_template("_job.html", job=data, return_to=mode),
            message=message,
        ), 202
    flash(message, "success")
    return redirect(url_for("dashboard.index", **({"mode": mode} if mode != "fixed" else {})))


def _send_result(filename: str, validator, directory, as_attachment: bool = False):
    try:
        filename = validator(filename)
    except (FileNotFoundError, ValueError):
        abort(404)
    return send_from_directory(directory.resolve(), filename, as_attachment=as_attachment)


@dashboard.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        mode = request.args.get("mode", "fixed")
        if mode not in {"fixed", "adversarial"}:
            abort(404)
        service = get_service()
        results = service.result_snapshot(mode)
        catalog = service.figure_builder.catalog(mode, results)
        if "context" not in request.args:
            if set(request.args) - {"mode"}:
                abort(400, description="incomplete dashboard query")
            default = default_browsing_query(catalog, mode)
            return render_template("index.html", **_experiment_context(
                mode, results=results, browsing=_browsing_navigation(bootstrap_query=default),
            ))
        try:
            query, requested_page, page_size = parse_browsing_query(request.args, catalog, mode)
            projection = project_dashboard_query(
                results, catalog, query,
                presentations=service.game_presentations if mode == "fixed" else {},
            )
        except ValueError as error:
            abort(400, description=str(error))
        page = paginate_projection(projection, requested_page, page_size)
        canonical = query_string(query, page.page, page.page_size)
        if request.query_string != canonical.encode("ascii"):
            return redirect(_browse_url(query, page.page, page.page_size))
        return render_template("index.html", **_experiment_context(
            mode, results=results, browse_page=page, browsing=_browsing_navigation(page),
        ))

    if request.form.get("experiment_type") == "adversarial":
        return _submit_one_player()

    try:
        form = _parse_form()
        job = get_service().submit_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return _form_error("fixed", get_service().default_form_state(), error)

    kind = "horizon batch" if len(form.horizon_values) > 1 else "experiment"
    return _queued_experiment_response(job, "fixed", f"Queued {kind} job {job.id[:8]}.")


@dashboard.get("/experiment-groups/fixed/<group_id>/players/<player>")
def fixed_group_detail(group_id: str, player: str):
    if re.fullmatch(r"[0-9a-f]{16}", group_id) is None:
        return jsonify(error="Invalid result group identifier."), 400
    if re.fullmatch(r"[0-9]+", player) is None:
        return jsonify(error="Invalid player for this result."), 400
    try:
        detail = fixed_group_detail_context(get_service(), group_id, int(player))
    except ValueError:
        return jsonify(error="Invalid player for this result."), 400
    except KeyError:
        return jsonify(error="This result is no longer available. Refresh results."), 404
    response = jsonify(detail)
    response.cache_control.no_store = True
    return response


def _submit_one_player():
    service = get_service()
    try:
        form = _parse_one_player_form(parse_adversarial_experiment_form, service)
        job = service.submit_adversarial_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        return _form_error("adversarial", service.default_adversarial_form_state(), error)

    kind = "one-player batch" if len(form.action_counts) > 1 or len(form.horizon_values) > 1 else "one-player experiment"
    return _queued_experiment_response(job, "adversarial", f"Queued {kind} job {job.id[:8]}.")


@dashboard.get("/adversarial/experiments/<filename>")
def download_adversarial_experiment(filename: str):
    service = get_service()
    return _send_result(filename, service.validate_adversarial_csv_filename, service.adversarial_raw_dir, True)


@dashboard.route("/custom-games", methods=["GET", "POST"])
def custom_games():
    service = get_service()
    if request.method == "POST":
        try:
            n_players = parse_positive_integer(request.form["n_players"], "number of players", MAX_CUSTOM_PLAYERS)
            payoff_structure = request.form.get("payoff_structure", "zero_sum")
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


@dashboard.post("/experiment-groups/<kind>/<group_id>/delete")
def delete_result_group(kind: str, group_id: str):
    redirect_arguments = {"mode": "adversarial"} if kind == "adversarial" else {}
    try:
        deleted = get_service().delete_result_group(kind, group_id)
    except (FileNotFoundError, KeyError, OSError, ServiceBusyError, ValueError) as error:
        flash(str(error), "error")
    else:
        flash(f"Deleted experiment group with {deleted} replicate file(s) and cleared generated artifacts.", "success")
    return redirect(url_for("dashboard.index", **redirect_arguments))


def _filtered_deletion_state() -> dict:
    values = request.form.to_dict(flat=True)
    values["profiles"] = request.form.getlist("profiles")
    return values


@dashboard.post("/experiment-groups/<kind>/delete-filtered/preview")
def preview_filtered_result_groups(kind: str):
    try:
        preview = get_service().preview_filtered_result_groups(kind, _filtered_deletion_state())
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(count=preview.count, digest=preview.membership_digest)


@dashboard.post("/experiment-groups/<kind>/delete-filtered")
def delete_filtered_result_groups(kind: str):
    try:
        deleted = get_service().delete_filtered_result_groups(
            kind, _filtered_deletion_state(), request.form.get("digest", ""),
        )
    except FilteredResultsChanged as error:
        return jsonify(error=str(error), count=error.count, reconfirmation_required=True), 409
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 400
    except (FileNotFoundError, OSError, ServiceBusyError) as error:
        return jsonify(error=str(error)), 409
    return jsonify(deleted=deleted, message=f"Deleted {deleted} filtered experiment(s).")


@dashboard.get("/figure-builder/options")
def figure_builder_options():
    try:
        return jsonify(get_service().figure_builder.catalog(request.args.get("mode", "fixed")))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@dashboard.post("/figure-builder/collection")
def build_figure_collection():
    try:
        result = get_service().figure_builder.build_collection(parse_profile_selection(request.form))
    except (ValueError, FileNotFoundError) as error:
        return jsonify(error=str(error)), 400
    return _figure_collection_response(result)


@dashboard.post("/figure-builder/cache")
def cached_figure_collection():
    try:
        result = get_service().figure_builder.cached_collection(parse_profile_selection(request.form))
    except (ValueError, FileNotFoundError) as error:
        return jsonify(error=str(error)), 400
    return _figure_collection_response(result)


def _figure_collection_response(result):
    for figure in result["figures"]:
        figure["url"] = url_for("dashboard.selected_figure", filename=figure["filename"])
        figure["pdf_url"] = url_for("dashboard.selected_figure", filename=figure["pdf_filename"])
    return jsonify(result)


@dashboard.get("/figure-builder/files/<filename>")
def selected_figure(filename: str):
    try:
        path = get_service().figure_builder.artifact_path(filename)
    except (ValueError, FileNotFoundError):
        abort(404)
    return send_file(path, as_attachment=path.suffix == ".pdf", download_name=path.name)


@dashboard.post("/figures/download-filtered.pdf")
def download_filtered_figures():
    service = get_service()
    mode = request.form.get("mode")
    if mode != "figure_builder":
        return jsonify(error="Unknown experiment mode."), 400
    directory = service.figure_builder.output_dir
    validator = lambda filename: service.figure_builder.artifact_path(filename).name
    filenames = request.form.getlist("filenames")
    if not filenames:
        return jsonify(error="No figures match the current filters."), 400

    paths = []
    try:
        for filename in filenames:
            path = (directory / validator(filename)).resolve()
            if path.parent != directory.resolve():
                raise ValueError("Figure is outside the result directory")
            paths.append(path)
    except (FileNotFoundError, ValueError):
        return jsonify(error="A selected figure is no longer available. Refresh the page and try again."), 404
    try:
        output = merged_figure_pdf(paths)
    except (OSError, ValueError, PyPdfError):
        return jsonify(error="Could not read a selected figure. Generate the figures again and retry."), 422

    response = send_file(
        output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="filtered-regret-figures.pdf",
        max_age=0,
    )
    response.cache_control.no_store = True
    return response


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
    except EquilibriumAnalysisUnavailable as error:
        return jsonify({"status": "unavailable", "error": str(error)}), 422
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


@dashboard.post("/reset")
def reset_results():
    return_to = request.form.get("return_to")
    redirect_arguments = {"mode": "adversarial"} if return_to == "adversarial" else {}
    if request.form.get("confirmation") != "reset-results":
        flash("Reset confirmation was missing.", "error")
        return redirect(url_for("dashboard.index", **redirect_arguments))

    try:
        get_service().clear_results()
    except ServiceBusyError as error:
        flash(str(error), "error")
    else:
        flash("Deleted all experiment-derived results, figures, and caches.", "success")
    return redirect(url_for("dashboard.index", **redirect_arguments))
