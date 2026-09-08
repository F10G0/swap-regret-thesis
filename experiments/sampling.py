"""Bounded downsampling that keeps new sparse and legacy dense runs aligned."""

from experiments.recording import recording_checkpoints


class CheckpointRows:
    def __init__(self, horizon: int, max_points: int):
        self.max_points = max_points
        self.targets = set(recording_checkpoints(horizon, max(2, max_points)))
        self.first_rows = []
        self.target_rows = []
        self.count = 0
        self.previous_time = None

    def add(self, row: dict) -> None:
        time = int(row["t"])
        if time != self.previous_time:
            self.count += 1
            self.previous_time = time
            if self.count > self.max_points:
                self.first_rows.clear()
        if self.count <= self.max_points:
            self.first_rows.append(row)
        if time in self.targets:
            self.target_rows.append(row)

    def rows(self) -> list[dict]:
        return self.first_rows if self.count <= self.max_points else self.target_rows
