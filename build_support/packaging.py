"""Resolve the immutable feature profile for a project build."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Mapping

from setuptools import find_namespace_packages


EXPERIMENTAL_TRAJECTORIES_VARIABLE = "EXPERIMENTAL_TRAJECTORIES"
EXPERIMENTAL_TRAJECTORIES_MARKER = (
    "swap_regret_experimental_trajectories_enabled"
)
EXPERIMENTAL_TRAJECTORIES_MARKER_DIR = Path(
    "build_support/experimental_trajectories_enabled/"
    "swap_regret_experimental_trajectories_enabled"
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_CORE_PACKAGE_PATTERNS = (
    "algorithms*",
    "environments*",
    "experiments*",
    "metrics*",
    "web*",
)


def parse_experimental_trajectories(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return the requested build profile; an unspecified profile is off."""
    source = environ if environment is None else environment
    raw_value = source.get(EXPERIMENTAL_TRAJECTORIES_VARIABLE, "0")
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{EXPERIMENTAL_TRAJECTORIES_VARIABLE} must be one of "
        "0/1, off/on, false/true, or no/yes"
    )


@dataclass(frozen=True)
class BuildConfiguration:
    experimental_trajectories: bool
    packages: tuple[str, ...]
    package_dir: dict[str, str]
    package_data: dict[str, list[str]]
    exclude_package_data: dict[str, list[str]]

    def setup_arguments(self) -> dict:
        return {
            "packages": list(self.packages),
            "package_dir": dict(self.package_dir),
            "package_data": {
                package: list(patterns)
                for package, patterns in self.package_data.items()
            },
            "exclude_package_data": {
                package: list(patterns)
                for package, patterns in self.exclude_package_data.items()
            },
        }


def build_configuration(
    environment: Mapping[str, str] | None = None,
) -> BuildConfiguration:
    enabled = parse_experimental_trajectories(environment)
    include = list(_CORE_PACKAGE_PATTERNS)
    if enabled:
        include.append("experimental*")

    packages = find_namespace_packages(
        where=".",
        include=include,
        exclude=["tests*"],
    )
    package_dir: dict[str, str] = {}
    package_data: dict[str, list[str]] = {
        "web": [
            "templates/*.html",
            "static/*.css",
            "static/*.js",
            "static/equilibria/*.png",
            "static/equilibria/*.pdf",
        ],
    }
    exclude_package_data: dict[str, list[str]] = {}

    if enabled:
        packages.append(EXPERIMENTAL_TRAJECTORIES_MARKER)
        package_dir[EXPERIMENTAL_TRAJECTORIES_MARKER] = str(
            EXPERIMENTAL_TRAJECTORIES_MARKER_DIR
        )
        package_data["experimental.equilibrium_trajectory"] = ["README.md"]
    else:
        exclude_package_data["web"] = [
            "templates/experimental_trajectory.html",
            "static/experimental_trajectory.js",
        ]

    return BuildConfiguration(
        experimental_trajectories=enabled,
        packages=tuple(packages),
        package_dir=package_dir,
        package_data=package_data,
        exclude_package_data=exclude_package_data,
    )
