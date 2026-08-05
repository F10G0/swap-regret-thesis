from pathlib import Path

import pytest

import web
from build_support.packaging import (
    EXPERIMENTAL_TRAJECTORIES_MARKER,
    build_configuration,
    parse_experimental_trajectories,
)
from web.services import DashboardService


def test_experimental_trajectories_are_off_by_default() -> None:
    assert parse_experimental_trajectories({}) is False
    configuration = build_configuration({})

    assert configuration.experimental_trajectories is False
    assert EXPERIMENTAL_TRAJECTORIES_MARKER not in configuration.packages
    assert not any(
        package == "experimental" or package.startswith("experimental.")
        for package in configuration.packages
    )
    assert configuration.exclude_package_data["web"] == [
        "templates/experimental_trajectory.html",
        "static/experimental_trajectory.js",
    ]


def test_enabled_build_contains_code_assets_and_marker() -> None:
    configuration = build_configuration(
        {"EXPERIMENTAL_TRAJECTORIES": "1"}
    )

    assert configuration.experimental_trajectories is True
    assert "experimental.equilibrium_trajectory" in configuration.packages
    assert EXPERIMENTAL_TRAJECTORIES_MARKER in configuration.packages
    assert Path(
        configuration.package_dir[EXPERIMENTAL_TRAJECTORIES_MARKER]
    ).is_dir()
    assert configuration.package_data[
        "experimental.equilibrium_trajectory"
    ] == ["README.md"]
    assert "static/equilibria/*.pdf" in configuration.package_data["web"]
    assert configuration.exclude_package_data == {}


@pytest.mark.parametrize(
    "value",
    ["1", "true", "yes", "on", " TRUE "],
)
def test_enabled_build_value_spellings(value: str) -> None:
    assert parse_experimental_trajectories(
        {"EXPERIMENTAL_TRAJECTORIES": value}
    ) is True


@pytest.mark.parametrize(
    "value",
    ["", "0", "false", "no", "off", " OFF "],
)
def test_disabled_build_value_spellings(value: str) -> None:
    assert parse_experimental_trajectories(
        {"EXPERIMENTAL_TRAJECTORIES": value}
    ) is False


def test_unknown_build_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="EXPERIMENTAL_TRAJECTORIES"):
        parse_experimental_trajectories(
            {"EXPERIMENTAL_TRAJECTORIES": "perhaps"}
        )


def test_disabled_build_has_no_trajectory_web_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web,
        "experimental_trajectories_built",
        lambda: False,
    )
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
        custom_game_dir=tmp_path / "custom-games",
    )
    app = web.create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        service=service,
    )
    client = app.test_client()

    dashboard_response = client.get("/")

    assert dashboard_response.status_code == 200
    assert b"Experimental trajectory comparison" not in (
        dashboard_response.data
    )
    assert client.get(
        "/experimental/trajectory-comparisons"
    ).status_code == 404
    assert client.get("/trajectory-comparisons").status_code == 404
    assert "experimental_trajectory.workspace" not in app.view_functions
    assert service._experimental_trajectory_dashboard is None


def test_test_override_cannot_enable_a_non_testing_app(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        web,
        "experimental_trajectories_built",
        lambda: False,
    )
    app = web.create_app({
        "SECRET_KEY": "test-secret",
        "RESULTS_DIR": tmp_path,
        "RAW_DIR": tmp_path / "raw",
        "FIGURE_DIR": tmp_path / "figures",
        "CUSTOM_GAME_DIR": tmp_path / "custom-games",
        "EXPERIMENTAL_TRAJECTORIES_ENABLED": True,
        "TEST_ENABLE_EXPERIMENTAL_TRAJECTORIES": True,
    })

    assert app.config["EXPERIMENTAL_TRAJECTORIES_ENABLED"] is False
    assert "experimental_trajectory.workspace" not in app.view_functions
