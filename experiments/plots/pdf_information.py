"""Plain experiment-information pages for self-contained figure PDFs."""

from io import BytesIO
from textwrap import wrap

from matplotlib.figure import Figure


def experiment_information_pdf(rows: list[tuple[str, str]]) -> BytesIO:
    """Return one fixed-layout PDF page containing ordered key-value rows."""
    label_width = max((len(label) for label, _ in rows), default=0) + 1
    lines = ["Experiment Information", ""]
    for label, value in rows:
        prefix = f"{label}:".ljust(label_width + 2)
        value_lines = str(value).splitlines() or [""]
        first = True
        for value_line in value_lines:
            wrapped = wrap(value_line, width=max(20, 92 - len(prefix)), break_long_words=False,
                           break_on_hyphens=False) or [""]
            for part in wrapped:
                lines.append((prefix if first else " " * len(prefix)) + part)
                first = False

    figure = Figure(figsize=(8.27, 11.69))
    font_size = min(10.0, max(6.0, 590.0 / max(len(lines), 1)))
    figure.text(0.08, 0.93, "\n".join(lines), va="top", family="DejaVu Sans Mono",
                fontsize=font_size, linespacing=1.35)
    output = BytesIO()
    figure.savefig(output, format="pdf")
    output.seek(0)
    return output
