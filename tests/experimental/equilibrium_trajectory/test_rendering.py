from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from metrics.equilibrium_convergence import EquilibriumDistanceTrajectory


def test_trajectory_subdivides_final_log_interval_while_distance_is_unchanged(tmp_path, monkeypatch) -> None:
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence
    from experiments.plots import plot_equilibrium_convergence as distance_plot

    path = run_full_information_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=1_000, seed=42, output_dir=tmp_path
    )
    distance_horizons = []
    trajectory_horizons = []
    fit_horizons = []
    rendered_trajectories = []

    def capture_distance(payoff_tensor, empirical):
        distance_horizons.append(empirical.horizons.tolist())
        zeros = np.zeros(len(empirical.horizons))
        return EquilibriumDistanceTrajectory(empirical.horizons, zeros, zeros)

    def capture_trajectory(
        payoff_tensor,
        empirical,
        support_query_cap,
        geometry=None,
        relative_render_tolerance=None,
        fit_empirical=None,
    ):
        trajectory_horizons.append(empirical.horizons.tolist())
        fit_horizons.append(fit_empirical.horizons.tolist())
        projection = SimpleNamespace(
            transform=lambda vectors: np.zeros((len(vectors), 2))
        )
        return SimpleNamespace(
            empirical=empirical,
            projected_trajectory=projection.transform(empirical.vectors),
            projection=projection,
            ce_region=None,
            cce_region=None,
        )

    def capture_trajectory_plot(
        analysis,
        output_path,
        game_name,
        n_replicates,
        focus_from_checkpoint=0,
    ):
        rendered_trajectories.append(
            (
                analysis.empirical.horizons.tolist(),
                focus_from_checkpoint,
            )
        )

    monkeypatch.setattr(distance_plot, "equilibrium_distance_trajectory", capture_distance)
    monkeypatch.setattr(plot_equilibrium_convergence, "project_equilibrium_trajectory", capture_trajectory)
    monkeypatch.setattr(distance_plot, "_plot_equilibrium_distance", lambda *args: None)
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_plot_equilibrium_trajectory",
        capture_trajectory_plot,
    )

    distance_plot.plot_result_equilibrium_distance(
        path,
        tmp_path / "distance.png",
    )
    plot_equilibrium_convergence.plot_result_equilibrium_trajectory(
        path,
        tmp_path / "trajectory.png",
        final_interval_segments=4,
    )
    plot_equilibrium_convergence.plot_result_equilibrium_trajectory(
        path,
        tmp_path / "trajectory-focused.png",
        final_interval_segments=4,
        focus_final_interval=True,
    )

    assert distance_horizons == [[1, 100, 1_000]]
    assert trajectory_horizons == [
        [1, 10, 100, 325, 550, 775, 1_000],
        [10, 100, 325, 550, 775, 1_000],
    ]
    assert fit_horizons == [
        [1, 10, 100, 325, 550, 775, 1_000],
        [100, 325, 550, 775, 1_000],
    ]
    assert rendered_trajectories == [
        ([1, 10, 100, 325, 550, 775, 1_000], 0),
        ([10, 100, 325, 550, 775, 1_000], 1),
    ]


def test_comparison_uses_shared_focused_checkpoints_and_retains_incoming_node(
    tmp_path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence

    first_path = run_full_information_cross_play_experiment(
        "rps",
        ["hedge", "hedge"],
        horizon=1_000,
        seed=42,
        output_dir=tmp_path,
    )
    second_path = run_full_information_cross_play_experiment(
        "rps",
        ["regret_matching", "regret_matching"],
        horizon=1_000,
        seed=42,
        output_dir=tmp_path,
    )
    captured = {}
    expected_analysis = SimpleNamespace(members=())

    def capture_comparison(
        payoff_tensor,
        empiricals,
        member_ids,
        support_query_cap,
        geometry,
        relative_render_tolerance,
        fit_empiricals,
    ):
        captured["member_ids"] = member_ids
        captured["empirical_horizons"] = [
            empirical.horizons.tolist() for empirical in empiricals
        ]
        captured["fit_horizons"] = [
            empirical.horizons.tolist() for empirical in fit_empiricals
        ]
        return expected_analysis

    def capture_plot(
        analysis,
        plot_members,
        output_path,
        game_name,
        focus_from_checkpoint,
    ):
        captured["analysis"] = analysis
        captured["plot_member_ids"] = [
            member.member_id for member in plot_members
        ]
        captured["focus_from_checkpoint"] = focus_from_checkpoint

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "project_equilibrium_trajectory_comparison",
        capture_comparison,
    )
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_plot_equilibrium_trajectory_comparison",
        capture_plot,
    )
    geometry_cache = SimpleNamespace(get=lambda *args: object())
    members = [
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "z-member",
            "Hedge",
            "#112233",
            (first_path,),
        ),
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "a-member",
            "RM",
            "#445566",
            (second_path,),
        ),
    ]

    returned = (
        plot_equilibrium_convergence
        .plot_result_equilibrium_trajectory_comparison(
            members,
            tmp_path / "comparison.png",
            final_interval_segments=4,
            focus_final_interval=True,
            geometry_cache=geometry_cache,
        )
    )

    assert returned is expected_analysis
    assert captured["member_ids"] == ["a-member", "z-member"]
    assert captured["plot_member_ids"] == ["a-member", "z-member"]
    assert captured["empirical_horizons"] == [
        [10, 100, 325, 550, 775, 1_000],
        [10, 100, 325, 550, 775, 1_000],
    ]
    assert captured["fit_horizons"] == [
        [100, 325, 550, 775, 1_000],
        [100, 325, 550, 775, 1_000],
    ]
    assert captured["focus_from_checkpoint"] == 1


def test_unified_comparison_focus_bypasses_geometry_and_keeps_predecessor(
    tmp_path,
    monkeypatch,
) -> None:
    from experiments.games import create_rock_paper_scissors_payoffs
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence

    payoffs = create_rock_paper_scissors_payoffs()
    first_profiles = [
        np.asarray([
            (position % 3, (position + 1) % 3)
            for position in range(1_000)
        ])
    ]
    second_profiles = [
        np.asarray([
            ((position + 1) % 3, (position + 2) % 3)
            for position in range(1_000)
        ])
    ]
    loaded = iter([
        ("rps", payoffs, first_profiles),
        ("rps", payoffs, second_profiles),
    ])
    captured = {}
    expected_analysis = SimpleNamespace(members=())

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "load_equilibrium_result_inputs",
        lambda *args, **kwargs: next(loaded),
    )

    def capture_unified(
        payoff_tensor,
        empiricals,
        member_ids,
        fit_empiricals,
    ):
        captured["member_ids"] = member_ids
        captured["empirical_horizons"] = [
            empirical.horizons.tolist() for empirical in empiricals
        ]
        captured["fit_horizons"] = [
            empirical.horizons.tolist() for empirical in fit_empiricals
        ]
        return expected_analysis

    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "project_unified_equilibrium_trajectory_comparison",
        capture_unified,
    )
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_plot_equilibrium_trajectory_comparison",
        lambda analysis, members, output_path, game_name, focus: (
            captured.update(focus=focus)
        ),
    )
    geometry_cache = SimpleNamespace(
        get=lambda *args: (_ for _ in ()).throw(
            AssertionError("unified view must not request affine geometry")
        )
    )
    members = [
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "b", "B", "#ff7f0e", ()
        ),
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "a", "A", "#1f77b4", ()
        ),
    ]

    returned = (
        plot_equilibrium_convergence
        .plot_result_equilibrium_trajectory_comparison(
            members,
            tmp_path / "unified.png",
            final_interval_segments=4,
            focus_final_interval=True,
            geometry_cache=geometry_cache,
            comparison_view="unified",
        )
    )

    assert returned is expected_analysis
    assert captured["member_ids"] == ["a", "b"]
    assert captured["empirical_horizons"] == [
        [10, 100, 325, 550, 775, 1_000],
        [10, 100, 325, 550, 775, 1_000],
    ]
    assert captured["fit_horizons"] == [
        [100, 325, 550, 775, 1_000],
        [100, 325, 550, 775, 1_000],
    ]
    assert captured["focus"] == 1


def test_comparison_renderer_excludes_incoming_node_from_shared_limits(
    tmp_path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence

    empiricals = [
        SimpleNamespace(horizons=np.array([10, 100, 550, 1_000])),
        SimpleNamespace(horizons=np.array([10, 100, 550, 1_000])),
    ]
    members = [
        SimpleNamespace(
            projected_trajectory=np.array([
                [-100.0, 100.0],
                [0.0, 0.2],
                [0.5, 0.1],
                [1.0, 0.0],
            ]),
            empirical=empiricals[0],
        ),
        SimpleNamespace(
            projected_trajectory=np.array([
                [100.0, 100.0],
                [0.2, 0.3],
                [0.7, 0.2],
                [1.2, 0.1],
            ]),
            empirical=empiricals[1],
        ),
    ]
    analysis = SimpleNamespace(
        members=members,
        ce_region=None,
        cce_region=None,
        view_kind="unified_equilibrium_relative",
        axis_labels=("shared x", "L1 distance to CCE"),
    )
    plot_members = [
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "a", "A", "#112233", ()
        ),
        plot_equilibrium_convergence.TrajectoryComparisonPlotMember(
            "b", "B", "#445566", ()
        ),
    ]
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_draw_region",
        lambda *args, **kwargs: None,
    )
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plot_equilibrium_convergence._plot_equilibrium_trajectory_comparison(
        analysis,
        plot_members,
        tmp_path / "comparison.png",
        "RPS",
        focus_from_checkpoint=1,
    )

    figure = plt.gcf()
    axes = figure.axes[0]
    assert axes.lines[0].get_xdata()[0] == -100.0
    assert axes.lines[1].get_xdata()[0] == 100.0
    assert axes.get_xlim()[0] > -1.0
    assert axes.get_xlim()[1] < 2.0
    assert axes.get_ylim()[0] == 0.0
    assert not axes.texts
    assert "Unified Equilibrium-Relative" in axes.get_title()
    close_figure(figure)


@pytest.mark.parametrize(
    ("round_number", "label"),
    [
        (1, "1"),
        (100, "100"),
        (1_234, "1k"),
        (15_000, "20k"),
        (149_000, "100k"),
        (999_999, "1m"),
    ],
)
def test_trajectory_endpoint_labels_round_to_one_significant_digit(
    round_number: int,
    label: str,
) -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _rounded_round_label,
    )

    assert _rounded_round_label(round_number) == label


def test_trajectory_labels_only_logarithmic_horizons_and_final() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _informative_horizon_positions,
    )

    assert _informative_horizon_positions(
        np.array([1, 10, 55, 100, 5_000])
    ) == [0, 1, 3, 4]


def test_trajectory_plot_has_larger_endpoint_labels_and_fixed_extent(
    tmp_path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence

    trajectory = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.2],
            [0.7, 0.6],
            [1.0, 0.8],
            [1.4, 1.0],
        ]
    )
    analysis = SimpleNamespace(
        cce_region=None,
        ce_region=None,
        projected_trajectory=trajectory,
        empirical=SimpleNamespace(
            horizons=np.array([1, 25_001, 50_001, 75_000, 100_000])
        ),
    )
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_draw_region",
        lambda *args, **kwargs: None,
    )
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)

    output_path = tmp_path / "trajectory.png"
    plot_equilibrium_convergence._plot_equilibrium_trajectory(
        analysis,
        output_path,
        "test",
        n_replicates=1,
    )

    figure = plt.gcf()
    axes = figure.axes[0]
    trajectory_lines = axes.lines
    assert [text.get_text() for text in axes.texts] == ["1", "100k"]
    assert all(text.get_fontsize() == 11 for text in axes.texts)
    assert len(trajectory_lines) == 1
    assert trajectory_lines[0].get_color() == "#4b5563"
    assert [collection.get_label() for collection in axes.collections] == [
        "Start",
        "End",
    ]
    assert [collection.get_sizes()[0] for collection in axes.collections] == [
        115,
        125,
    ]
    assert np.allclose(axes.get_xlim(), [-0.168, 1.568])
    assert np.allclose(axes.get_ylim(), [-0.12, 1.12])
    assert output_path.is_file()
    close_figure(figure)


def test_focus_final_interval_keeps_only_preceding_log_segment_offscreen(
    tmp_path,
    monkeypatch,
) -> None:
    from experimental.equilibrium_trajectory import rendering as plot_equilibrium_convergence

    trajectory = np.array(
        [
            [-100.0, 100.0],
            [-50.0, 50.0],
            [-10.0, 10.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )
    analysis = SimpleNamespace(
        cce_region=None,
        ce_region=None,
        projected_trajectory=trajectory,
        empirical=SimpleNamespace(
            horizons=np.array([1, 10, 100, 1_000, 5_000, 10_000])
        ),
        view_kind="equilibrium_relative",
        axis_labels=(
            "CE-relative CCE tangent coordinate",
            "L1 distance to CCE",
        ),
    )
    monkeypatch.setattr(
        plot_equilibrium_convergence,
        "_draw_region",
        lambda *args, **kwargs: None,
    )
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plot_equilibrium_convergence._plot_equilibrium_trajectory(
        analysis,
        tmp_path / "focused.png",
        "test",
        n_replicates=1,
        focus_from_checkpoint=3,
    )

    figure = plt.gcf()
    axes = figure.axes[0]
    assert axes.lines[0].get_xdata()[0] == -10.0
    assert axes.lines[0].get_ydata()[0] == 10.0
    assert np.allclose(axes.get_xlim(), [-0.12, 1.12])
    assert np.allclose(axes.get_ylim(), [0.0, 1.12])
    assert np.allclose(axes.collections[0].get_offsets()[0], [0.0, 0.0])
    assert [text.get_text() for text in axes.texts] == ["1k", "10k"]
    assert np.allclose(axes.texts[0].xy, [0.0, 0.0])
    assert axes.get_ylabel() == "L1 distance to CCE"
    assert axes.get_aspect() == "auto"
    close_figure(figure)


def test_view_includes_ce_landmark_without_fitting_entire_cce() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _equilibrium_view_limits,
    )

    trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
    ce_region = SimpleNamespace(
        affine_dimension=0,
        boundary=np.array([[3.0, 3.0]]),
    )
    cce_region = SimpleNamespace(
        affine_dimension=1,
        boundary=np.array([[3.0, 3.0], [5.0, 3.0]]),
    )

    lower, upper = _equilibrium_view_limits(
        trajectory,
        ce_region,
        cce_region,
    )

    assert np.allclose(lower, [-0.12, -0.12])
    assert np.all(upper > [3.0, 3.0])
    assert upper[0] < 5.0


def test_view_uses_nearest_ce_line_point_not_full_line() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _equilibrium_view_limits,
    )

    trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
    line_region = SimpleNamespace(
        affine_dimension=1,
        boundary=np.array([[3.0, 0.0], [3.0, 10.0]]),
    )

    lower, upper = _equilibrium_view_limits(
        trajectory,
        line_region,
        line_region,
    )

    assert lower[0] == pytest.approx(-0.12)
    assert upper[0] > 3.0
    assert upper[1] < 2.0


def test_view_adds_nearest_cce_anchor_when_cce_is_offscreen() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _equilibrium_view_limits,
    )

    trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
    ce_region = SimpleNamespace(
        affine_dimension=0,
        boundary=np.array([[0.5, 0.5]]),
    )
    cce_region = SimpleNamespace(
        affine_dimension=1,
        boundary=np.array([[4.0, 0.5], [5.0, 0.5]]),
    )

    lower, upper = _equilibrium_view_limits(
        trajectory,
        ce_region,
        cce_region,
    )

    assert np.allclose(lower, [-0.12, -0.12])
    assert upper[0] > 4.0
    assert upper[0] < 5.0
    assert upper[1] == pytest.approx(1.12)


def test_cce_containing_trajectory_view_does_not_broaden_limits() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _equilibrium_view_limits,
    )

    trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
    cce_region = SimpleNamespace(
        affine_dimension=2,
        boundary=np.array([
            [-10.0, -10.0],
            [10.0, -10.0],
            [10.0, 10.0],
            [-10.0, 10.0],
        ]),
    )

    lower, upper = _equilibrium_view_limits(
        trajectory,
        None,
        cce_region,
    )

    assert np.allclose(lower, [-0.12, -0.12])
    assert np.allclose(upper, [1.12, 1.12])


def test_outer_cce_marks_remain_visible_behind_ce() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _draw_region,
    )

    figure, axes = plt.subplots()
    point_region = SimpleNamespace(
        affine_dimension=0,
        boundary=np.array([[0.0, 0.0]]),
    )

    _draw_region(
        axes,
        point_region,
        "#60a5fa",
        "Projected CCE",
        0.2,
        outer=True,
    )
    _draw_region(
        axes,
        point_region,
        "#f59e0b",
        "Projected CE",
        0.3,
    )

    assert axes.collections[0].get_sizes()[0] == 100
    assert axes.collections[1].get_sizes()[0] == 165
    assert len(axes.collections[1].get_facecolors()) == 0
    assert not axes.collections[1].get_clip_on()
    assert axes.collections[1].get_zorder() > axes.collections[0].get_zorder()
    plt.close(figure)


def test_projected_ce_line_and_region_use_foreground_landmarks() -> None:
    from experimental.equilibrium_trajectory.rendering import (
        _draw_region,
    )

    figure, axes = plt.subplots()
    line_region = SimpleNamespace(
        affine_dimension=1,
        boundary=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    polygon_region = SimpleNamespace(
        affine_dimension=2,
        boundary=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]),
    )

    _draw_region(
        axes,
        line_region,
        "#f59e0b",
        "Projected CE line",
        0.3,
    )
    _draw_region(
        axes,
        polygon_region,
        "#f59e0b",
        "Projected CE region",
        0.3,
    )

    assert axes.lines[0].get_linestyle() == "--"
    assert axes.lines[0].get_zorder() == 6
    assert axes.patches[0].get_hatch() == "///"
    assert axes.lines[1].get_linestyle() == "--"
    assert axes.lines[1].get_zorder() == 6
    plt.close(figure)
