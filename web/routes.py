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

from web.services import DashboardService, PlotUpdateError, ServiceBusyError
from web.validation import ExperimentForm, parse_experiment_form


dashboard = Blueprint("dashboard", __name__)


def get_service() -> DashboardService:
    return current_app.extensions["dashboard_service"]


def _figure_data() -> list[dict]:
    return [{**figure, "url": url_for("dashboard.serve_figure", filename=figure["filename"])} for figure in get_service().figure_records()]


def _dashboard_context(form_state: dict | None = None, inline_error: str | None = None) -> dict:
    service = get_service()
    results = service.result_snapshot()
    jobs = service.jobs.recent()
    group_fields = ("game", "feedback_mode", "algorithm_player_0", "algorithm_player_1", "horizon", "seed", "stationary_method")
    replicates_by_group = {}
    for summary in results.summaries:
        key = tuple(summary[field] for field in group_fields)
        replicates_by_group.setdefault(key, set()).add(summary["replicate"])
    summaries = [
        {
            **summary,
            "replicate_count": len(replicates_by_group[tuple(summary[field] for field in group_fields)]),
            "download_url": url_for("dashboard.download_experiment", filename=summary["experiment"]),
            "joint_actions_url": url_for("dashboard.joint_actions", filename=summary["experiment"]),
        }
        for summary in results.summaries
    ]

    job_data = [
        {
            **job.public_data(),
            "url": url_for("dashboard.job_status", job_id=job.id),
        }
        for job in jobs
    ]
    return {
        "games": service.games,
        "feedback_modes": service.feedback_modes,
        "algorithms_by_feedback_mode": service.algorithms_by_feedback_mode,
        "algorithm_labels": service.algorithm_labels,
        "experiments": results.filenames,
        "figures": _figure_data(),
        "summaries": summaries,
        "warnings": results.warnings,
        "jobs": job_data,
        "busy": service.jobs.is_busy(),
        "form_state": form_state or service.default_form_state(),
        "inline_error": inline_error,
        "max_horizon": current_app.config["MAX_HORIZON"],
        "max_replicates": current_app.config["MAX_REPLICATES"],
    }


def _parse_form() -> ExperimentForm:
    service = get_service()
    return parse_experiment_form(
        request.form,
        games=set(service.games),
        algorithms_by_feedback_mode=service.algorithms_by_feedback_mode,
        max_horizon=current_app.config["MAX_HORIZON"],
        max_replicates=current_app.config["MAX_REPLICATES"],
    )


@dashboard.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", **_dashboard_context())

    try:
        form = _parse_form()
        job = get_service().submit_experiment(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        raw_state = get_service().default_form_state() | dict(request.form)
        return (
            render_template(
                "index.html",
                **_dashboard_context(raw_state, str(error)),
            ),
            400,
        )

    flash(f"Queued experiment job {job.id[:8]}.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard.post("/run-all-pairs")
def run_all_pairs():
    try:
        form = _parse_form()
        job, scheduled_count, skipped_count = get_service().submit_all_pairs(form)
    except (FileExistsError, ServiceBusyError, ValueError) as error:
        raw_state = get_service().default_form_state() | dict(request.form)
        return (
            render_template(
                "index.html",
                **_dashboard_context(raw_state, str(error)),
            ),
            400,
        )

    flash(f"Queued job {job.id[:8]} for {scheduled_count} runs; {skipped_count} already exist.", "success")
    return redirect(url_for("dashboard.index"))


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
    return jsonify(_figure_data())


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
    return redirect(url_for("dashboard.index"))


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


@dashboard.get("/experiments/<filename>/joint-actions.png")
def joint_actions(filename: str):
    try:
        path = get_service().joint_action_figure(filename)
    except (FileNotFoundError, KeyError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name)


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
        flash("Deleted generated CSV, PNG, and report files.", "success")
    return redirect(url_for("dashboard.index"))
