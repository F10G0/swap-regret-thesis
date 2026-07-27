from dataclasses import asdict, dataclass
from operator import index
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

import numpy as np

from config import CUSTOM_GAME_DIR
from experiments.games import PAYOFF_FACTORIES


CUSTOM_GAME_PREFIX = "custom__"
CUSTOM_GAME_FORMAT_VERSION = 1
MAX_CUSTOM_PLAYERS = 8
MAX_CUSTOM_ACTIONS_PER_PLAYER = 100
MAX_CUSTOM_PAYOFF_VALUES = 1_000_000
_SAFE_GAME_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class GameDefinition:
    id: str
    label: str
    description: str
    source: str
    n_players: int
    action_counts: tuple[int, ...]
    seed: int | None = None

    def public_data(self) -> dict:
        return asdict(self)


def _validated_custom_name(name: str) -> tuple[str, str]:
    label = " ".join(str(name).strip().split())
    slug = label.lower().replace(" ", "-")
    if not label or not _SAFE_GAME_NAME.fullmatch(slug):
        raise ValueError("game name must use 1-64 letters, numbers, spaces, hyphens, or underscores")
    return label, slug


def _validated_player_count(value) -> int:
    try:
        n_players = index(value)
    except TypeError as error:
        raise ValueError("number of players must be an integer") from error
    if not 2 <= n_players <= MAX_CUSTOM_PLAYERS:
        raise ValueError(f"number of players must be between 2 and {MAX_CUSTOM_PLAYERS}")
    return n_players


def _validated_action_counts(values, n_players: int) -> tuple[int, ...]:
    try:
        action_counts = tuple(index(value) for value in values)
    except TypeError as error:
        raise ValueError("action counts must be integers") from error
    if len(action_counts) != n_players:
        raise ValueError("provide exactly one action count per player")
    if any(count <= 0 or count > MAX_CUSTOM_ACTIONS_PER_PLAYER for count in action_counts):
        raise ValueError(f"each action count must be between 1 and {MAX_CUSTOM_ACTIONS_PER_PLAYER}")
    payoff_values = n_players * int(np.prod(action_counts, dtype=object))
    if payoff_values > MAX_CUSTOM_PAYOFF_VALUES:
        raise ValueError(f"payoff tensor must contain at most {MAX_CUSTOM_PAYOFF_VALUES:,} values")
    return action_counts


def _validated_seed(value) -> int:
    try:
        seed = index(value)
    except TypeError as error:
        raise ValueError("seed must be an integer") from error
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


class GameCatalog:
    def __init__(self, custom_game_dir: str | Path = CUSTOM_GAME_DIR):
        self.custom_game_dir = Path(custom_game_dir)
        self._write_lock = Lock()
        self._definition_cache: dict[Path, tuple[int, int, GameDefinition]] = {}
        self._built_in_definitions: dict[str, GameDefinition] | None = None

    def _custom_path(self, slug: str) -> Path:
        return self.custom_game_dir / f"{slug}.npz"

    def custom_path(self, game_id: str) -> Path:
        if not game_id.startswith(CUSTOM_GAME_PREFIX):
            raise ValueError(f"unknown custom game: {game_id}")
        slug = game_id.removeprefix(CUSTOM_GAME_PREFIX)
        if not _SAFE_GAME_NAME.fullmatch(slug):
            raise ValueError(f"unknown custom game: {game_id}")
        path = self._custom_path(slug)
        if not path.is_file():
            raise FileNotFoundError(f"custom game {game_id} does not exist")
        definition, _ = self._load_custom_file(path)
        if definition.id != game_id:
            raise ValueError(f"unknown custom game: {game_id}")
        return path

    def _load_custom_file(self, path: Path) -> tuple[GameDefinition, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            required = {"format_version", "name", "slug", "seed", "action_counts", "payoff_tensor"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
            if int(archive["format_version"]) != CUSTOM_GAME_FORMAT_VERSION:
                raise ValueError("unsupported format version")
            label, slug = _validated_custom_name(str(archive["name"]))
            if slug != str(archive["slug"]) or path.name != f"{slug}.npz":
                raise ValueError("file name does not match stored game name")
            seed = _validated_seed(int(archive["seed"]))
            payoff_tensor = np.asarray(archive["payoff_tensor"], dtype=float)
            n_players = _validated_player_count(payoff_tensor.shape[0] if payoff_tensor.ndim else 0)
            action_counts = _validated_action_counts(tuple(int(value) for value in archive["action_counts"]), n_players)

        if payoff_tensor.shape != (n_players, *action_counts):
            raise ValueError("payoff tensor shape does not match its metadata")
        if not np.all(np.isfinite(payoff_tensor)) or np.any((payoff_tensor < 0.0) | (payoff_tensor > 1.0)):
            raise ValueError("payoffs must be finite values in [0, 1]")
        description = f"Custom random game · {n_players} players · actions {' × '.join(map(str, action_counts))} · seed {seed}"
        definition = GameDefinition(f"{CUSTOM_GAME_PREFIX}{slug}", label, description, "custom", n_players, action_counts, seed)
        return definition, payoff_tensor

    def custom_definitions(self) -> tuple[list[GameDefinition], list[str]]:
        if not self.custom_game_dir.exists():
            return [], []
        definitions = []
        warnings = []
        active_paths = set()
        for path in sorted(self.custom_game_dir.glob("*.npz")):
            active_paths.add(path)
            try:
                stat = path.stat()
                cached = self._definition_cache.get(path)
                if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
                    definition = cached[2]
                else:
                    definition, _ = self._load_custom_file(path)
                    self._definition_cache[path] = (stat.st_mtime_ns, stat.st_size, definition)
            except (OSError, TypeError, ValueError) as error:
                warnings.append(f"Skipped {path.name}: {error}")
            else:
                definitions.append(definition)
        self._definition_cache = {path: value for path, value in self._definition_cache.items() if path in active_paths}
        return definitions, warnings

    def definitions(self) -> dict[str, GameDefinition]:
        if self._built_in_definitions is None:
            self._built_in_definitions = {}
            for game_id, factory in PAYOFF_FACTORIES.items():
                action_counts = tuple(factory().shape[1:])
                self._built_in_definitions[game_id] = GameDefinition(
                    game_id,
                    game_id.replace("_", " ").title(),
                    "",
                    "builtin",
                    len(action_counts),
                    action_counts,
                )
        definitions = dict(self._built_in_definitions)
        custom_definitions, _ = self.custom_definitions()
        definitions.update((definition.id, definition) for definition in custom_definitions)
        return definitions

    def load(self, game_id: str) -> np.ndarray:
        if game_id in PAYOFF_FACTORIES:
            return PAYOFF_FACTORIES[game_id]()
        if not game_id.startswith(CUSTOM_GAME_PREFIX):
            raise ValueError(f"unknown game: {game_id}")
        slug = game_id.removeprefix(CUSTOM_GAME_PREFIX)
        if not _SAFE_GAME_NAME.fullmatch(slug):
            raise ValueError(f"unknown game: {game_id}")
        path = self._custom_path(slug)
        if not path.is_file():
            raise ValueError(f"unknown game: {game_id}")
        definition, payoff_tensor = self._load_custom_file(path)
        if definition.id != game_id:
            raise ValueError(f"unknown game: {game_id}")
        return payoff_tensor

    def create_random(self, name: str, n_players, action_counts, seed) -> GameDefinition:
        label, slug = _validated_custom_name(name)
        n_players = _validated_player_count(n_players)
        action_counts = _validated_action_counts(action_counts, n_players)
        seed = _validated_seed(seed)
        payoff_tensor = np.random.default_rng(seed).random((n_players, *action_counts))
        output_path = self._custom_path(slug)

        with self._write_lock:
            self.custom_game_dir.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                raise FileExistsError(f"custom game {label!r} already exists")
            with tempfile.TemporaryDirectory(prefix=".custom-game-", dir=self.custom_game_dir) as temporary_directory:
                temporary_path = Path(temporary_directory) / output_path.name
                np.savez_compressed(
                    temporary_path,
                    format_version=np.array(CUSTOM_GAME_FORMAT_VERSION),
                    name=np.array(label),
                    slug=np.array(slug),
                    seed=np.array(seed),
                    action_counts=np.asarray(action_counts),
                    payoff_tensor=payoff_tensor,
                )
                os.replace(temporary_path, output_path)

        definition, _ = self._load_custom_file(output_path)
        return definition

    def delete(self, game_id: str) -> GameDefinition:
        with self._write_lock:
            path = self.custom_path(game_id)
            definition, _ = self._load_custom_file(path)
            path.unlink()
            self._definition_cache.pop(path, None)
        return definition


def load_game_payoffs(game_id: str, custom_game_dir: str | Path = CUSTOM_GAME_DIR) -> np.ndarray:
    return GameCatalog(custom_game_dir).load(game_id)
