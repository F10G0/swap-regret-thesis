"""Lifecycle cleanup for artifacts derived from experiment results."""

from argparse import ArgumentParser
import os
from pathlib import Path
from collections.abc import Iterable


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def clear_experiment_artifacts(
    roots: Iterable[str | Path],
    *,
    preserve: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Remove generated content below ``roots``, retaining inputs and placeholders.

    Caller-supplied roots remain in place. Protected paths may be inside a root;
    existing ``.gitkeep`` files are always retained.
    """
    protected = tuple(_resolved(path) for path in preserve)
    removed: list[Path] = []
    seen_roots: set[Path] = set()

    def protected_path(path: Path) -> bool:
        resolved = _resolved(path)
        return any(_is_within(resolved, item) for item in protected)

    def protected_directory(path: Path) -> bool:
        resolved = _resolved(path)
        return any(
            _is_within(resolved, item) or _is_within(item, resolved)
            for item in protected
        )

    for root_value in roots:
        root = Path(root_value)
        resolved_root = _resolved(root)
        if resolved_root in seen_roots or not root.is_dir() or root.is_symlink():
            continue
        seen_roots.add(resolved_root)

        for directory_name, directory_names, filenames in os.walk(
            root, topdown=False, followlinks=False
        ):
            directory = Path(directory_name)
            for filename in filenames:
                path = directory / filename
                if filename == ".gitkeep" or protected_path(path):
                    continue
                path.unlink()
                removed.append(path)
            for child_name in directory_names:
                child = directory / child_name
                if protected_directory(child):
                    continue
                if child.is_symlink():
                    child.unlink()
                    removed.append(child)
                else:
                    try:
                        child.rmdir()
                    except OSError:
                        pass

    return tuple(removed)


def main() -> None:
    parser = ArgumentParser(description="Remove all artifacts derived from experiment results.")
    parser.add_argument("--preserve", action="append", default=[], help="source/configuration path to retain (may be repeated)")
    parser.add_argument("roots", nargs="+", help="generated artifact roots")
    arguments = parser.parse_args()
    clear_experiment_artifacts(arguments.roots, preserve=arguments.preserve)


if __name__ == "__main__":
    main()
