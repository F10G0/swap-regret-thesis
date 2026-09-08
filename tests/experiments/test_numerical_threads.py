import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("override", [None, "2"])
def test_make_configures_numerical_threads_before_python_launch(override):
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    environment = {key: value for key, value in os.environ.items() if key not in names}
    if override is not None:
        environment.update(dict.fromkeys(names, override))
    script = f"import json, os; print(json.dumps([os.environ.get(name) for name in {names!r}]))"
    target = f"numerical-threads: ; @$(PYTHON) -c \"{script}\""
    result = subprocess.run(
        ["make", "--no-print-directory", f"PYTHON={sys.executable}", "--eval", target, "numerical-threads"],
        cwd=Path(__file__).resolve().parents[2], env=environment, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == [override or "1"] * len(names)
