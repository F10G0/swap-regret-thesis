import csv

import pytest

from experiments.recorder import CsvRecorder


def test_recorder_commits_atomically(tmp_path) -> None:
    output_path = tmp_path / "run.csv"

    with CsvRecorder(["value"], output_path) as recorder:
        recorder.record({"value": 1})

    with output_path.open("r", encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == [{"value": "1"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_recorder_removes_partial_output_after_failure(tmp_path) -> None:
    output_path = tmp_path / "run.csv"

    with pytest.raises(RuntimeError, match="interrupted"):
        with CsvRecorder(["value"], output_path) as recorder:
            recorder.record({"value": 1})
            raise RuntimeError("interrupted")

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
