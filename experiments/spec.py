from dataclasses import dataclass
from hashlib import sha256
import json

from config import STATIONARY_METHOD


FEEDBACK_MODES = {"full_information", "bandit"}


@dataclass(frozen=True)
class ExperimentSpec:
    game_name: str
    feedback_mode: str
    algorithm_names: tuple[str, ...]
    horizon: int
    seed: int
    replicate: int = 0
    stationary_method: str = STATIONARY_METHOD

    def __post_init__(self) -> None:
        if self.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"unknown feedback mode: {self.feedback_mode}")
        if len(self.algorithm_names) != 2:
            raise ValueError("exactly two player algorithms are required")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        if not self.stationary_method:
            raise ValueError("stationary_method must not be empty")

    @property
    def algorithm_profile_name(self) -> str:
        return "_vs_".join(self.algorithm_names)

    @property
    def run_id(self) -> str:
        payload = json.dumps(self.configuration(), sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()[:8]
        return f"{self.game_name}_{self.feedback_mode}_{self.algorithm_profile_name}_{digest}"

    def configuration(self) -> dict:
        return {
            "game_name": self.game_name,
            "feedback_mode": self.feedback_mode,
            "algorithm_names": self.algorithm_names,
            "horizon": self.horizon,
            "seed": self.seed,
            "replicate": self.replicate,
            "stationary_method": self.stationary_method,
        }

    def metadata(self) -> dict:
        return {
            "run_id": self.run_id,
            "feedback_mode": self.feedback_mode,
            "seed": self.seed,
            "replicate": self.replicate,
            "stationary_method": self.stationary_method,
            "algorithm_player_0": self.algorithm_names[0],
            "algorithm_player_1": self.algorithm_names[1],
            "horizon": self.horizon,
        }
