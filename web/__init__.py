import os
from pathlib import Path
import secrets

from flask import Flask, abort, request, session

from config import CUSTOM_GAME_DIR, FIGURE_DIR, RAW_DIR, RESULTS_DIR
from web.build_features import experimental_trajectories_built
from web.routes import dashboard
from web.services import DashboardService


def create_app(
    test_config: dict | None = None,
    service: DashboardService | None = None,
) -> Flask:
    experimental_trajectories_enabled = experimental_trajectories_built()
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SWAP_REGRET_WEB_SECRET") or secrets.token_hex(32),
        RESULTS_DIR=RESULTS_DIR,
        RAW_DIR=RAW_DIR,
        FIGURE_DIR=FIGURE_DIR,
        CUSTOM_GAME_DIR=CUSTOM_GAME_DIR,
        MAX_HORIZON=100_000,
        MAX_REPLICATES=100,
        EXPERIMENTAL_TRAJECTORIES_ENABLED=experimental_trajectories_enabled,
    )

    if test_config:
        app.config.update(test_config)

    if service is None:
        service = DashboardService(
            results_dir=Path(app.config["RESULTS_DIR"]),
            raw_dir=Path(app.config["RAW_DIR"]),
            figure_dir=Path(app.config["FIGURE_DIR"]),
            custom_game_dir=Path(app.config["CUSTOM_GAME_DIR"]),
        )

    app.extensions["dashboard_service"] = service
    app.register_blueprint(dashboard)
    if app.config.get("TESTING") and app.config.get("TEST_ENABLE_EXPERIMENTAL_TRAJECTORIES"):
        experimental_trajectories_enabled = True
    app.config["EXPERIMENTAL_TRAJECTORIES_ENABLED"] = experimental_trajectories_enabled
    if experimental_trajectories_enabled:
        from web.experimental_routes import experimental_trajectory

        app.register_blueprint(experimental_trajectory)

    @app.context_processor
    def inject_csrf_token() -> dict:
        def csrf_token() -> str:
            token = session.get("_csrf_token")
            if token is None:
                token = secrets.token_urlsafe(32)
                session["_csrf_token"] = token
            return token

        return {
            "csrf_token": csrf_token,
            "experimental_trajectories_enabled": experimental_trajectories_enabled,
        }

    @app.before_request
    def protect_post_requests() -> None:
        if request.method != "POST":
            return

        expected_token = session.get("_csrf_token", "")
        submitted_token = request.form.get("_csrf_token", "")
        if not expected_token or not secrets.compare_digest(
            expected_token,
            submitted_token,
        ):
            abort(400, description="invalid or missing CSRF token")

    return app


__all__ = ["create_app"]
