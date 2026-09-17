import importlib
import json
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from setuptools import find_packages

from experiments.runtime_environment import runtime_environment_json
from tests.web.support import create_test_app, dashboard_data


def test_core_packages_import_from_the_current_project():
    root = Path.cwd().resolve()
    for name in ("algorithms", "environments", "experiments", "metrics", "web"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(root)


def test_dashboard_entry_points_and_retired_static_subsystem(tmp_path):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    assert set(dashboard_data(client.get("/"))["gameDefinitions"]) == {"matching_pennies", "rps", "rpsls"}
    assert client.get("/?mode=adversarial").status_code == 200
    assert client.get("/custom-games").status_code == 200
    custom = service.create_custom_game("Local", 2, [2, 2], 42, "zero_sum")
    assert b"Profile Weight" not in client.get(f"/custom-games/{custom.id}").data
    assert not any("/equilibria/" in rule.rule for rule in app.url_map.iter_rules())
    assert not (Path("web/static") / "equilibria").exists()
    assert "equilibriumFigures" not in dashboard_data(client.get("/"))


def test_production_imports_and_distance_solver_need_no_retired_solver():
    # Intentional dependency names: fail if the retired stack is imported again.
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

class NoRetiredSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'games_learning', 'pulp'}:
            raise AssertionError('retired solver imported: ' + fullname)

sys.meta_path.insert(0, NoRetiredSolver())
for name in ('algorithms', 'environments', 'experiments', 'metrics', 'web'):
    package = importlib.import_module(name)
    for module in pkgutil.walk_packages(package.__path__, name + '.'):
        importlib.import_module(module.name)

import numpy as np
from experiments.game_catalog import load_game_payoffs
from metrics.equilibrium_distance import equilibrium_l1_distance
for game, size in [('matching_pennies', 2), ('rps', 3), ('rpsls', 5)]:
    for concept in ('ce', 'cce'):
        result = equilibrium_l1_distance(load_game_payoffs(game), np.full((size, size), 1 / size**2), concept)
        assert abs(result.distance) < 1e-9
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


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
    assert set(json.loads(runtime_environment_json())["packages"]) == {"matplotlib", "numpy", "scipy"}
