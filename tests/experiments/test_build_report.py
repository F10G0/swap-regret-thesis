from experiments.build_report import build_report


def test_build_report_creates_output_directory_and_lists_figures(tmp_path) -> None:
    figure_dir = tmp_path / "results" / "figures"
    figure_dir.mkdir(parents=True)
    (figure_dir / "regret.png").write_bytes(b"png")

    output_path = build_report(figure_dir, tmp_path / "report")

    assert output_path == tmp_path / "report" / "index.html"
    report = output_path.read_text(encoding="utf-8")
    assert "../results/figures/regret.png" in report
    assert "regret" in report
