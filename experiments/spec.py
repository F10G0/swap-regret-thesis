from dataclasses import dataclass
from hashlib import sha256
import json
import re

from config import STATIONARY_METHOD
from experiments.result_schema import resolve_regret_evaluation


FEEDBACK_MODES = {"full_information", "bandit"}
MAX_RUN_ID_BYTES = 200
ALGORITHM_RUN_ID_TOKENS = {
    "regret_matching": "rm",
    "stationary_regret_matching": "srm",
}
PAYOFF_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ExperimentSpec:
    game_name: str
    feedback_mode: str
    algorithm_names: tuple[str, ...]
    horizon: int
    seed: int
    replicate: int = 0
    stationary_method: str = STATIONARY_METHOD
    regret_evaluation: str = "feedback_aligned"
    game_payoff_digest: str = ""

    def __post_init__(self) -> None:
        if self.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"unknown feedback mode: {self.feedback_mode}")
        if len(self.algorithm_names) < 2:
            raise ValueError("at least two player algorithms are required")
        if any(not name for name in self.algorithm_names):
            raise ValueError("algorithm names must not be empty")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        object.__setattr__(self, "regret_evaluation", resolve_regret_evaluation(self.feedback_mode, self.regret_evaluation))
        if not self.stationary_method:
            raise ValueError("stationary_method must not be empty")
        if self.game_payoff_digest and not PAYOFF_DIGEST_PATTERN.fullmatch(self.game_payoff_digest):
            raise ValueError("game_payoff_digest must be a lowercase SHA-256 digest")

    @property
    def algorithm_profile_name(self) -> str:
        return "_vs_".join(self.algorithm_names)

    @property
    def abbreviated_algorithm_profile_name(self) -> str:
        return "_vs_".join(ALGORITHM_RUN_ID_TOKENS.get(name, name) for name in self.algorithm_names)

    @property
    def run_id(self) -> str:
        payload = json.dumps(self.configuration(), sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()
        readable_run_id = f"{self.game_name}_{self.feedback_mode}_{self.algorithm_profile_name}_{digest[:8]}"
        if len(readable_run_id.encode("utf-8")) <= MAX_RUN_ID_BYTES:
            return readable_run_id
        abbreviated_run_id = f"{self.game_name}_{self.feedback_mode}_{self.abbreviated_algorithm_profile_name}_{digest[:8]}"
        if len(abbreviated_run_id.encode("utf-8")) <= MAX_RUN_ID_BYTES:
            return abbreviated_run_id
        compact_run_id = f"{self.game_name}_{self.feedback_mode}_{len(self.algorithm_names)}p_{digest[:16]}"
        if len(compact_run_id.encode("utf-8")) <= MAX_RUN_ID_BYTES:
            return compact_run_id
        return f"experiment_{len(self.algorithm_names)}p_{digest[:16]}"

    def configuration(self) -> dict:
        return {
            "game_name": self.game_name,
            "feedback_mode": self.feedback_mode,
            "algorithm_names": self.algorithm_names,
            "horizon": self.horizon,
            "seed": self.seed,
            "replicate": self.replicate,
            "regret_evaluation": self.regret_evaluation,
            "stationary_method": self.stationary_method,
            "game_payoff_digest": self.game_payoff_digest,
        }

    def metadata(self) -> dict:
        return {
            "run_id": self.run_id,
            "feedback_mode": self.feedback_mode,
            "regret_evaluation": self.regret_evaluation,
            "seed": self.seed,
            "replicate": self.replicate,
            "stationary_method": self.stationary_method,
            "game_payoff_digest": self.game_payoff_digest,
            "algorithm_profile": json.dumps(self.algorithm_names, separators=(",", ":")),
            "horizon": self.horizon,
        }
