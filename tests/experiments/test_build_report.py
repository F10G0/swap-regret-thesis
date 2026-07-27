import csv
import json
from pathlib import Path

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


def _write_result(
    path: Path,
) -> None:
    fieldnames = (
        BASE_FIELDNAMES
        + EXPECTED_FIELDS
    )

    common = {
        "run_id": "test-run",
        "feedback_mode":
            "full_information",
        "regret_evaluation":
            "expected",
        "seed": "42",
        "replicate": "0",
        "stationary_method":
            "solve",
        "game": "rps",
        "algorithm":
            "hedge_vs_hedge",
        "n_players": "2",
        "algorithm_profile":
            json.dumps(
                ["hedge", "hedge"]
            ),
        "horizon": "10",
        "t": "10",
    }

    rows = []

    for player in range(2):
        rows.append({
            **common,

            "player_algorithm":
                "hedge",

            "algorithm_player_0":
                "hedge",

            "algorithm_player_1":
                "hedge",

            "player":
                str(player),

            "action":
                str(player),

            "payoff":
                "0.5",

            "expected_external_regret":
                "1.0",

            "average_expected_external_regret":
                "0.1",

            "expected_internal_regret":
                "0.8",

            "average_expected_internal_regret":
                "0.08",

            "expected_swap_regret":
                "1.2",

            "average_expected_swap_regret":
                "0.12",
        })

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def test_build_report_is_static_and_interactive(
    tmp_path: Path,
) -> None:
    results_dir = (
        tmp_path
        / "results"
    )

    raw_dir = (
        results_dir
        / "raw"
    )

    figure_dir = (
        results_dir
        / "figures"
    )

    details_dir = (
        figure_dir
        / "details"
    )

    equilibrium_dir = (
        tmp_path
        / "equilibria"
    )

    raw_dir.mkdir(
        parents=True
    )

    details_dir.mkdir(
        parents=True
    )

    equilibrium_dir.mkdir(
        parents=True
    )

    result_path = (
        raw_dir
        / "test-run.csv"
    )

    _write_result(
        result_path
    )

    regret_figure = (
        figure_dir
        / (
            "rps_average_expected_"
            "external_regret_player_0.png"
        )
    )

    regret_figure.write_bytes(
        b"test"
    )

    detail_figure = (
        details_dir
        / (
            "test-run_"
            "equilibrium_distance.png"
        )
    )

    detail_figure.write_bytes(
        b"test"
    )

    equilibrium_figure = (
        equilibrium_dir
        / (
            "rps_ce_blue_lower_origin_"
            "maximum_profile_weight.png"
        )
    )

    equilibrium_figure.write_bytes(
        b"test"
    )

    output_path = build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=raw_dir,
        equilibrium_dir=equilibrium_dir,
    )

    assert output_path == (
        results_dir
        / "index.html"
    )

    html = output_path.read_text(
        encoding="utf-8"
    )

    assert (
        "__REPORT_DATA__"
        not in html
    )

    assert (
        "Static interactive snapshot"
        in html
    )

    assert (
        "Hedge vs Hedge"
        in html
    )

    assert (
        "rps_average_expected_"
        "external_regret_player_0.png"
        in html
    )

    assert (
        "test-run_"
        "equilibrium_distance.png"
        in html
    )

    assert (
        "rps_ce_blue_lower_origin_"
        "maximum_profile_weight.png"
        in html
    )

    assert "fetch(" not in html

    copied_equilibrium = (
        results_dir
        / "figures"
        / "report"
        / "equilibria"
        / equilibrium_figure.name
    )

    assert copied_equilibrium.is_file()


def test_build_report_handles_empty_results(
    tmp_path: Path,
) -> None:
    results_dir = (
        tmp_path
        / "results"
    )

    figure_dir = (
        results_dir
        / "figures"
    )

    equilibrium_dir = (
        tmp_path
        / "equilibria"
    )

    figure_dir.mkdir(
        parents=True
    )

    equilibrium_dir.mkdir(
        parents=True
    )

    output_path = build_report(
        figure_dir=figure_dir,
        results_dir=results_dir,
        raw_dir=results_dir / "raw",
        equilibrium_dir=equilibrium_dir,
    )

    assert output_path.is_file()

    html = output_path.read_text(
        encoding="utf-8"
    )

    assert (
        "Swap Regret Experiment Report"
        in html
    )
