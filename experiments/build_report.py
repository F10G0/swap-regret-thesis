from pathlib import Path

from config import FIGURE_DIR, RESULTS_DIR


def main() -> None:
    figure_paths = sorted(
        FIGURE_DIR.glob("*.png")
    )

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Swap Regret Experiments</title>",
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
        }

        h1 {
            margin-bottom: 40px;
        }

        .figure {
            margin-bottom: 50px;
        }

        img {
            max-width: 100%;
            border: 1px solid #ccc;
        }

        .caption {
            margin-top: 10px;
            font-family: monospace;
        }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>Swap Regret Experiment Report</h1>",
    ]

    for figure_path in figure_paths:
        relative_path = figure_path.relative_to(RESULTS_DIR)

        html.extend(
            [
                '<div class="figure">',
                f'<img src="{relative_path}" alt="{figure_path.name}">',
                f'<div class="caption">{figure_path.stem}</div>',
                "</div>",
            ]
        )

    html.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    output_path = RESULTS_DIR / "index.html"
    output_path.write_text(
        "\n".join(html),
        encoding="utf-8",
    )

    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
