import csv
import os
from pathlib import Path
import tempfile


class CsvRecorder:
    def __init__(self, fieldnames: list[str], output_path: str | Path):
        self.fieldnames = fieldnames
        self.output_path = Path(output_path)

    def __enter__(self) -> "CsvRecorder":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            dir=self.output_path.parent,
            delete=False,
        )
        self.temporary_path = Path(temporary_file.name)
        self.file = temporary_file
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="raise")
            self.writer.writeheader()
        except BaseException:
            self.file.close()
            self.temporary_path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.file.flush()
                os.fsync(self.file.fileno())
            self.file.close()
            if exc_type is None:
                os.link(self.temporary_path, self.output_path)
        finally:
            if not self.file.closed:
                self.file.close()
            self.temporary_path.unlink(missing_ok=True)

    def record(self, row: dict) -> None:
        self.writer.writerow(row)
