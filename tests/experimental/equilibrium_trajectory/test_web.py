from pathlib import Path

from experiments.scenarios.full_information_cross_play import (
    run_full_information_cross_play_experiment,
)
from web import create_app
from web.services import DashboardService


def _app_with_result(tmp_path: Path):
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
        custom_game_dir=tmp_path / "custom-games",
    )
    service._publish_plots = lambda game_name=None: None
    run_full_information_cross_play_experiment(
        "rps",
        ["hedge", "hedge"],
        horizon=1,
        output_dir=service.raw_dir,
    )
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "TEST_ENABLE_EXPERIMENTAL_TRAJECTORIES": True,
        },
        service=service,
    ), service


def test_trajectory_workspace_is_explicit_and_not_on_core_dashboard(
    tmp_path: Path,
) -> None:
    app, service = _app_with_result(tmp_path)
    client = app.test_client()

    core_page = client.get("/")

    assert core_page.status_code == 200
    assert b'id="trajectory-comparison-view"' not in core_page.data
    assert service._experimental_trajectory_dashboard is None

    experimental_page = client.get(
        "/experimental/trajectory-comparisons"
    )

    assert experimental_page.status_code == 200
    assert b'id="trajectory-comparison-view"' in experimental_page.data
    assert b"experimental_trajectory.js" in experimental_page.data
    assert service._experimental_trajectory_dashboard is not None


def test_view_change_only_marks_pending_experimental_state() -> None:
    script = Path("web/static/experimental_trajectory.js").read_text(
        encoding="utf-8"
    )
    handler = script.split(
        'listen("trajectory-comparison-view", "change",', 1
    )[1].split("});", 1)[0]

    assert "saveTrajectoryComparisonView" in handler
    assert "updateTrajectoryComparisonDirtyState" in handler
    assert "generateTrajectoryComparison" not in handler
