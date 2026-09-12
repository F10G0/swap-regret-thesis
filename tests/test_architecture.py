import importlib
from pathlib import Path
import subprocess
import sys

from tests.web.support import create_test_app


def test_core_packages_import_from_the_current_project():
    root = Path.cwd().resolve()
    for name in ("algorithms", "environments", "experiments", "metrics", "web"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(root)


def test_dashboard_has_one_application_blueprint(tmp_path):
    app, _ = create_test_app(tmp_path)
    assert set(app.blueprints) == {"dashboard"}
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    assert all(endpoint == "static" or endpoint.startswith("dashboard.") for endpoint in endpoints)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/?mode=adversarial").status_code == 200
    assert client.get("/custom-games").status_code == 200


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
for game, size in [('rps', 3), ('rpsls', 5)]:
    for concept in ('ce', 'cce'):
        result = equilibrium_l1_distance(load_game_payoffs(game), np.full((size, size), 1 / size**2), concept)
        assert abs(result.distance) < 1e-9
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
