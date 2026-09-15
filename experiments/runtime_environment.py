"""Canonical identity for the numerical runtime used to create results."""

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
import platform


_NUMERICAL_DISTRIBUTIONS = (
    "matplotlib",
    "numpy",
    "scipy",
)


@lru_cache(maxsize=1)
def runtime_environment_json() -> str:
    packages = {}
    for distribution_name in _NUMERICAL_DISTRIBUTIONS:
        try:
            installed = distribution(distribution_name)
        except PackageNotFoundError:
            packages[distribution_name] = {"version": "missing"}
            continue
        identity = {"version": installed.version}
        direct_url = installed.read_text("direct_url.json")
        if direct_url:
            try:
                vcs_info = json.loads(direct_url).get("vcs_info", {})
            except (AttributeError, json.JSONDecodeError):
                vcs_info = {}
            commit = vcs_info.get("commit_id")
            if commit:
                identity["vcs_commit"] = commit
        packages[distribution_name] = identity

    lock_path = Path(__file__).resolve().parents[1] / "requirements.lock"
    lock_fingerprint = (
        sha256(lock_path.read_bytes()).hexdigest()
        if lock_path.is_file()
        else "unavailable"
    )
    environment = {
        "lock_sha256": lock_fingerprint,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": packages,
    }
    return json.dumps(environment, sort_keys=True, separators=(",", ":"))


def validate_runtime_environment(serialized: str) -> str:
    try:
        environment = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("runtime_environment must be valid JSON") from error
    if not isinstance(environment, dict):
        raise ValueError("runtime_environment must be a JSON object")
    return json.dumps(environment, sort_keys=True, separators=(",", ":"))


def runtime_environment_fingerprint(canonical: str) -> str:
    """Hash runtime JSON already canonicalized at construction or file input."""
    digest = sha256()
    digest.update(b"swap-regret-runtime-environment-v1\0")
    digest.update(canonical.encode("utf-8"))
    return digest.hexdigest()
