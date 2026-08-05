import ast
from pathlib import Path
import subprocess
import sys


CORE_PATHS = (
    Path("algorithms"),
    Path("environments"),
    Path("experiments"),
    Path("metrics"),
    Path("main.py"),
)


def _python_files(path: Path):
    if path.is_file():
        yield path
    else:
        yield from path.rglob("*.py")


def test_core_modules_do_not_import_experimental_subsystem() -> None:
    violations = []
    for root in CORE_PATHS:
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    if module == "experimental" or module.startswith(
                        "experimental."
                    ):
                        violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_normal_dashboard_startup_does_not_import_trajectory_package() -> None:
    source = (
        "import sys; import web.app; "
        "assert not any(name == 'experimental' or "
        "name.startswith('experimental.') for name in sys.modules), "
        "sorted(name for name in sys.modules if "
        "name.startswith('experimental'))"
    )
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        cwd=Path.cwd(),
    )
