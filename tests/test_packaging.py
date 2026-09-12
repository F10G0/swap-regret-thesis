import json
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from setuptools import find_packages

from experiments.runtime_environment import runtime_environment_json


def test_static_package_discovery_covers_only_core_packages():
    with Path("pyproject.toml").open("rb") as file:
        settings = tomllib.load(file)["tool"]["setuptools"]
    discovery = settings["packages"]["find"]
    packages = find_packages(include=discovery["include"], exclude=discovery["exclude"])
    assert {name.split(".")[0] for name in packages} == {
        "algorithms", "environments", "experiments", "metrics", "web",
    }
    assert set(packages) == set(find_packages(exclude=["tests*"]))
    assert settings["py-modules"] == ["config"]
    assert discovery["namespaces"] is False


def test_static_package_data_covers_all_normal_web_assets():
    with Path("pyproject.toml").open("rb") as file:
        settings = tomllib.load(file)["tool"]["setuptools"]
    web = Path("web")
    included = {path for pattern in settings["package-data"]["web"] for path in web.glob(pattern)}
    required = {path for directory in (web / "templates", web / "static")
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in {".html", ".css", ".js", ".png", ".pdf"}}
    assert included == required


def test_dependency_configuration_and_fingerprint_exclude_retired_solvers():
    # These names are intentional regression guards, not runtime dependencies.
    retired = ("games_learning", "games-learning", "pulp")
    with Path("pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    requirements = project["project"]["dependencies"] + Path("requirements.lock").read_text().splitlines()
    assert not any(name in requirement.lower() for requirement in requirements for name in retired)
    for filename in ("requirements.txt", "Makefile", ".gitignore", "pyrightconfig.json"):
        assert not any(name in Path(filename).read_text().lower() for name in retired)
    assert project["tool"]["setuptools"]["package-data"]["web"] == [
        "templates/*.html", "static/*.css", "static/*.js",
    ]
    assert set(json.loads(runtime_environment_json())["packages"]) == {"matplotlib", "numpy", "scipy"}
