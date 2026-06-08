import csv
from pathlib import Path


class CsvRecorder:
    def __init__(self, fieldnames: list[str]):
        self.fieldnames = fieldnames
        self.rows: list[dict] = []

    def record(self, row: dict) -> None:
        self.rows.append(row)

    def save(self, output_path: str | Path) -> None:
        if not self.rows:
            raise ValueError("no rows to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(self.rows)

    def clear(self) -> None:
        self.rows.clear()
