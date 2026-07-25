import csv
from pathlib import Path


class CsvRecorder:
    def __init__(self, fieldnames: list[str], output_path: str | Path):
        self.fieldnames = fieldnames
        self.output_path = Path(output_path)

    def __enter__(self) -> "CsvRecorder":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
        self.file = self.temporary_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="raise")
        self.writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.file.close()
        if exc_type is None:
            self.temporary_path.replace(self.output_path)
        else:
            self.temporary_path.unlink(missing_ok=True)

    def record(self, row: dict) -> None:
        self.writer.writerow(row)
