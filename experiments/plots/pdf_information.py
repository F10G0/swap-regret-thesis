"""Plain experiment-information pages for self-contained figure PDFs."""

from io import BytesIO
from textwrap import wrap

from matplotlib.figure import Figure
import numpy as np


ENDPOINT_STATISTICS_DESCRIPTION = "replicate mean, sample SD (ddof=1), SE=SD/sqrt(n)"


def summarize_values(values) -> tuple[float, float | None, float | None]:
    """Return the arithmetic mean, sample SD, and SE for finite values."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("summary values must be a non-empty finite one-dimensional sequence")
    mean = float(np.mean(data))
    if len(data) == 1:
        return mean, None, None
    sample_sd = float(np.std(data, ddof=1))
    return mean, sample_sd, sample_sd / np.sqrt(len(data))


def format_value_summary(values) -> str:
    mean, sample_sd, standard_error = summarize_values(values)
    if sample_sd is None:
        return f"mean = {mean:.6g}, SD = n/a, SE = n/a"
    return f"mean = {mean:.6g}, SD = {sample_sd:.6g}, SE = {standard_error:.6g}"


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
