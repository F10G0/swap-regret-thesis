import csv
import os
from pathlib import Path
import tempfile


def require_csv_columns(input_path: str | Path, fieldnames, required) -> None:
    missing = set(required) - set(fieldnames)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {', '.join(sorted(missing))}")


def read_final_csv_rows(input_path: str | Path, group_column: str | None = None) -> tuple[set[str], list[dict[str, str]]]:
    input_path = Path(input_path)
    with input_path.open("rb") as file:
        header = file.readline()
        if not header:
            return set(), []
        header_text = header.decode("utf-8")
        fieldnames = set(next(csv.reader([header_text]), []))
        if group_column is not None and group_column not in fieldnames:
            return fieldnames, []
        data_start = file.tell()
        file.seek(0, 2)
        position = file.tell()
        buffer = b""

        while True:
            chunk_start = max(data_start, position - 8192)
            file.seek(chunk_start)
            buffer = file.read(position - chunk_start) + buffer
            position = chunk_start
            lines = buffer.splitlines()
            if position > data_start and lines:
                lines = lines[1:]
            if lines:
                rows = list(csv.DictReader([header_text, *(line.decode("utf-8") for line in lines)]))
                if group_column is None:
                    return fieldnames, rows[-1:]
                final_value = rows[-1][group_column]
                first_final = len(rows) - 1
                while first_final > 0 and rows[first_final - 1][group_column] == final_value:
                    first_final -= 1
                if first_final > 0 or position == data_start:
                    return fieldnames, rows[first_final:]
            if position == data_start:
                return fieldnames, []


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
