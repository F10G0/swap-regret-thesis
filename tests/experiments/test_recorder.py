import pytest

from experiments.recorder import CsvRecorder, read_final_csv_rows
from tests.support import read_csv_rows


def test_recorder_commits_atomically(tmp_path) -> None:
    output_path = tmp_path / "run.csv"

    with CsvRecorder(["value"], output_path) as recorder:
        recorder.record({"value": 1})

    assert read_csv_rows(output_path) == [{"value": "1"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_recorder_removes_partial_output_after_failure(tmp_path) -> None:
    output_path = tmp_path / "run.csv"

    with pytest.raises(RuntimeError, match="interrupted"):
        with CsvRecorder(["value"], output_path) as recorder:
            recorder.record({"value": 1})
            raise RuntimeError("interrupted")

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_recorders_publish_without_overwriting_each_other(tmp_path) -> None:
    output_path = tmp_path / "run.csv"

    with pytest.raises(FileExistsError):
        with CsvRecorder(["value"], output_path) as first:
            first.record({"value": "first"})
            with CsvRecorder(["value"], output_path) as second:
                second.record({"value": "second"})

    assert read_csv_rows(output_path) == [{"value": "second"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_final_csv_reader_returns_the_complete_last_group_across_chunks(tmp_path) -> None:
    output_path = tmp_path / "grouped.csv"
    with CsvRecorder(["group", "payload"], output_path) as recorder:
        for row in range(300):
            recorder.record({"group": "final" if row >= 297 else str(row), "payload": "x" * 50})

    fieldnames, final_group = read_final_csv_rows(output_path, "group")
    _, final_row = read_final_csv_rows(output_path)

    assert fieldnames == {"group", "payload"}
    assert len(final_group) == 3
    assert {row["group"] for row in final_group} == {"final"}
    assert final_row == final_group[-1:]
