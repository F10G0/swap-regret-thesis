import importlib
from pathlib import Path

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
