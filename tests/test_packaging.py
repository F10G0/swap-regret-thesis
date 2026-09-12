from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from setuptools import find_packages


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
