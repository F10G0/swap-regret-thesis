from html import escape
import os
from pathlib import Path

from config import FIGURE_DIR, RESULTS_DIR


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Swap Regret Experiments</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1 {{ margin-bottom: 40px; }}
.figure {{ margin-bottom: 50px; }}
img {{ max-width: 100%; border: 1px solid #ccc; }}
.caption {{ margin-top: 10px; font-family: monospace; }}
</style>
</head>
<body>
<h1>Swap Regret Experiment Report</h1>
{figures}
</body>
</html>
"""


def build_report(figure_dir: str | Path = FIGURE_DIR, results_dir: str | Path = RESULTS_DIR) -> Path:
    figure_dir = Path(figure_dir)
    results_dir = Path(results_dir)
    figures = "\n".join(
        f'<div class="figure"><img src="{escape(Path(os.path.relpath(path, results_dir)).as_posix())}" alt="{escape(path.name)}"><div class="caption">{escape(path.stem)}</div></div>'
        for path in sorted(figure_dir.glob("*.png"))
    )
    output_path = results_dir / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(HTML_TEMPLATE.format(figures=figures), encoding="utf-8")
    return output_path


def main() -> None:
    output_path = build_report()
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
