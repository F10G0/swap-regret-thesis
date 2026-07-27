import csv
import json
from pathlib import Path

import experiments.build_report as report_module
from experiments.build_report import build_report
from experiments.result_schema import BASE_FIELDNAMES


EXPECTED_FIELDS = [
    "expected_external_regret",
    "average_expected_external_regret",
    "expected_internal_regret",
    "average_expected_internal_regret",
    "expected_swap_regret",
    "average_expected_swap_regret",
]


def _write_result(path: Path) -> None:
    fieldnames = BASE_FIELDNAMES + EXPECTED_FIELDS

    common = {
        "run_id": "test-run",
        "feedback_mode": "full_information",
        "regret_evaluation": "expected",
        "seed": "42",
        "replicate": "0",
        "stationary_method": "solve",
        "game": "rps",
        "algorithm": "hedge_vs_hedge",
        "n_players": "2",
        "algorithm_profile": json.dumps(["hedge", "hedge"]),
        "horizon": "10",
        "t": "10",
    }

    rows = []
    for player in range(2):
        rows.append(
            {
                **common,
                "player_algorithm": "hedge",
                "algorithm_player_0": "hedge",
                "algorithm_player_1": "hedge",
                "player": str(player),
                "action": str(player),
                "payoff": "0.5",
                "expected_external_regret": "1.0",
                "average_expected_external_regret": "0.1",
                "expected_internal_regret": "0.8",
                "average_expected_internal_regret": "0.08",
                "expected_swap_regret": "1.2",
                "average_expected_swap_regret": "0.12",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fake_plot(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake png")


def test_build_report_is_static_and_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    raw_dir = results_dir / "raw"
    figure_dir = results_dir / "figures"
    equilibrium_dir = tmp_path / "equilibria"

    raw_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    equilibrium_dir.mkdir(parents=True)

    _write_result(raw_dir / "test-run.csv")

    # Existing top-level regret figure.
    regret_figure = (
        figure_dir
        / "rps_average_expected_external_regret_player_0.png"
    )
    regret_figure.write_bytes(b"fake png")

    # Precomputed theoretical equilibrium figure.
    equilibrium_figure = (
        equilibrium_dir
        / "rps_ce_blue_lower_origin_maximum_profile_weight.png"
    )
    equilibrium_figure.write_bytes(b"fake png")

    joint_calls = []
    distance_calls = []
    trajectory_calls = []

    def fake_joint_actions(input_paths, output_path) -> None:
        joint_calls.append(list(input_paths))
        _fake_plot(output_path)

    def fake_distance(input_paths, output_path, **kwargs) -> None:
        distance_calls.append(list(input_paths))
        _fake_plot(output_path)

    def fake_trajectory(
        input_paths,
        output_path,
        trajectory_points=10,
        hide_first=False,
        **kwargs,
    ) -> None:
        trajectory_calls.append(
            {
                "paths": list(input_paths),
                "trajectory_points": trajectory_points,
                "hide_first": hide_first,
            }
        )
        _fake_plot(output_path)

    # Patch the names imported directly into experiments.build_report.
    monkeypatch.setattr(
        report_module,
        "plot_joint_actions",
        fake_joint_actions,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_distance",
        fake_distance,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_trajectory",
        fake_trajectory,
    )

    output_path = build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=raw_dir,
        equilibrium_dir=equilibrium_dir,
    )

    assert output_path == results_dir / "index.html"
    assert output_path.is_file()

    html = output_path.read_text(encoding="utf-8")

    # General static-report behavior.
    assert "__REPORT_DATA__" not in html
    assert "Static interactive snapshot" in html
    assert "Hedge vs Hedge" in html
    assert "fetch(" not in html

    # Existing regret figure is included.
    assert (
        "rps_average_expected_external_regret_player_0.png"
        in html
    )

    # Theoretical equilibrium figure is copied into the report.
    copied_equilibrium = (
        results_dir
        / "figures"
        / "report"
        / "equilibria"
        / equilibrium_figure.name
    )
    assert copied_equilibrium.is_file()
    assert equilibrium_figure.name in html

    # Group-level analysis is generated exactly once where appropriate.
    assert len(joint_calls) == 1
    assert len(distance_calls) == 1

    # Both static trajectory variants are generated.
    assert len(trajectory_calls) == 2
    assert {
        (
            call["trajectory_points"],
            call["hide_first"],
        )
        for call in trajectory_calls
    } == {
        (10, False),
        (10, True),
    }

    # Both trajectory variants are present in the generated HTML.
    assert (
        "_p10_from_round_1_"
        "replicate_mean_equilibrium_trajectory.png"
        in html
    )
    assert (
        "_p10_hide_round_1_"
        "replicate_mean_equilibrium_trajectory.png"
        in html
    )
    
    # Static UI exposes and wires the Hide first toggle.
    assert 'toggleText.textContent = "Hide first"' in html
    assert 'checkbox.addEventListener("change", setTrajectory)' in html


def test_build_report_reuses_current_detail_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    raw_dir = results_dir / "raw"
    figure_dir = results_dir / "figures"
    equilibrium_dir = tmp_path / "equilibria"

    raw_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    equilibrium_dir.mkdir(parents=True)

    _write_result(raw_dir / "test-run.csv")

    # First build generates the detail figures with cheap fake plotters.
    def fake_joint_actions(input_paths, output_path) -> None:
        _fake_plot(output_path)

    def fake_distance(input_paths, output_path, **kwargs) -> None:
        _fake_plot(output_path)

    def fake_trajectory(input_paths, output_path, **kwargs) -> None:
        _fake_plot(output_path)

    monkeypatch.setattr(
        report_module,
        "plot_joint_actions",
        fake_joint_actions,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_distance",
        fake_distance,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_trajectory",
        fake_trajectory,
    )

    build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=raw_dir,
        equilibrium_dir=equilibrium_dir,
    )

    # A second build should reuse current caches rather than regenerate them.
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "current detail figure should have been reused"
        )

    monkeypatch.setattr(
        report_module,
        "plot_joint_actions",
        fail_if_called,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_distance",
        fail_if_called,
    )
    monkeypatch.setattr(
        report_module,
        "plot_result_equilibrium_trajectory",
        fail_if_called,
    )

    output_path = build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=raw_dir,
        equilibrium_dir=equilibrium_dir,
    )

    assert output_path.is_file()


def test_build_report_handles_empty_results(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    figure_dir = results_dir / "figures"
    equilibrium_dir = tmp_path / "equilibria"

    figure_dir.mkdir(parents=True)
    equilibrium_dir.mkdir(parents=True)

    output_path = build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=results_dir / "raw",
        equilibrium_dir=equilibrium_dir,
    )

    assert output_path.is_file()

    html = output_path.read_text(encoding="utf-8")

    assert "Swap Regret Experiment Report" in html
    assert "No matching summaries." in html
