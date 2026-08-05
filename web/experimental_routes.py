"""HTTP surface registered only when trajectory support was built."""

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from web.services import DashboardService


experimental_trajectory = Blueprint("experimental_trajectory", __name__)


def get_service() -> DashboardService:
    return current_app.extensions["dashboard_service"]


@experimental_trajectory.get("/trajectory-comparisons")
def trajectory_comparison():
    from experimental.equilibrium_trajectory.settings import (
        parse_final_interval_segments,
        parse_focus_final_interval,
        parse_trajectory_comparison_view,
    )

    try:
        group_ids = request.args.getlist("member")
        final_interval_segments = parse_final_interval_segments(
            request.args.get("final_interval_segments")
        )
        focus_final_interval = parse_focus_final_interval(
            request.args.get("focus_final_interval")
        )
        comparison_view = parse_trajectory_comparison_view(
            request.args.get("comparison_view")
        )
    except ValueError as error:
        abort(400, description=str(error))
    try:
        result, error = get_service().experimental_trajectory.request(
            group_ids,
            final_interval_segments,
            focus_final_interval,
            comparison_view,
        )
    except ValueError as error:
        abort(400, description=str(error))
    except (FileNotFoundError, KeyError):
        abort(404)
    if error is not None:
        return jsonify({"status": "failed", "error": error}), 500
    if result is None:
        response = jsonify({
            "status": "generating",
            "message": "Computing shared trajectory comparison…",
        })
        response.status_code = 202
        response.headers["Retry-After"] = "2"
        return response
    image_url = url_for(
        "experimental_trajectory.trajectory_comparison_artifact",
        artifact_id=result.definition.artifact_id,
        figure_format="png",
    )
    pdf_url = url_for(
        "experimental_trajectory.trajectory_comparison_artifact",
        artifact_id=result.definition.artifact_id,
        figure_format="pdf",
    )
    return jsonify(result.public_data(image_url, pdf_url))


@experimental_trajectory.get("/trajectory-comparisons/<artifact_id>.<figure_format>")
def trajectory_comparison_artifact(artifact_id: str, figure_format: str):
    try:
        path = get_service().experimental_trajectory.artifact(artifact_id, figure_format)
    except (FileNotFoundError, ValueError):
        abort(404)
    return send_from_directory(path.parent.resolve(), path.name)


@experimental_trajectory.get("/experimental/trajectory-comparisons")
def workspace():
    from experimental.equilibrium_trajectory.settings import (
        DEFAULT_FINAL_INTERVAL_SEGMENTS,
        MAX_FINAL_INTERVAL_SEGMENTS,
        MIN_FINAL_INTERVAL_SEGMENTS,
    )

    service = get_service()
    return render_template(
        "experimental_trajectory.html",
        trajectory_comparison_candidates=(
            service.experimental_trajectory.candidates()
        ),
        trajectory_comparison_url=url_for(
            "experimental_trajectory.trajectory_comparison"
        ),
        game_presentations=service.game_presentations,
        default_final_interval_segments=(
            DEFAULT_FINAL_INTERVAL_SEGMENTS
        ),
        min_final_interval_segments=MIN_FINAL_INTERVAL_SEGMENTS,
        max_final_interval_segments=MAX_FINAL_INTERVAL_SEGMENTS,
    )
