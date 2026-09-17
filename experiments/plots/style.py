"""Shared thesis-sized plotting defaults; no experiment or metric semantics."""

from functools import wraps
from textwrap import fill
from threading import RLock

import matplotlib as mpl
import numpy as np


FIGURE_WIDTH = 6.3
MARKER_STEP = 0.24
LINE_FIGURE_SIZE = (FIGURE_WIDTH, 3.8)
HEATMAP_FIGURE_SIZE = (FIGURE_WIDTH, 4.7)
ANNOTATION_SIZE = 9
_STYLE_LOCK = RLock()
PUBLICATION_RC = {
    "figure.figsize": LINE_FIGURE_SIZE,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.7,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "savefig.dpi": 150,
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PROFILE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#AA4499", "#E69F00", "#56B4E9", "#882255", "#444444")
PROFILE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
REGRET_STYLES = {
    "external": ("#0072B2", "-", "o"),
    "internal": ("#CC79A7", "-.", "s"),
    "swap": ("#D55E00", "--", "^"),
}


def publication_plot(function):
    """Apply typography during artist creation and saving, restoring rcParams."""
    @wraps(function)
    def styled(*args, **kwargs):
        with _STYLE_LOCK, mpl.rc_context(PUBLICATION_RC):
            return function(*args, **kwargs)
    return styled


def staggered_markevery(index: int, count: int) -> tuple[float, float]:
    return index / count * MARKER_STEP, MARKER_STEP


def profile_series_style(index: int, profile_count: int) -> dict:
    return dict(color=PROFILE_COLORS[index % len(PROFILE_COLORS)], linestyle="-",
                marker=PROFILE_MARKERS[index % len(PROFILE_MARKERS)], markevery=staggered_markevery(index, profile_count))


def regret_series_style(name: str, index: int = 0, count: int = 1) -> dict:
    color, linestyle, marker = REGRET_STYLES[name]
    return dict(color=color, linestyle=linestyle, marker=marker, markevery=staggered_markevery(index, count),
                zorder=3 if name == "swap" else 2)


def regret_axis_label(kind: str, view: str = "average") -> str:
    numerator = rf"R_T^{{\mathrm{{{kind}}}}}"
    suffix = {"average": "/T", "sqrt_scaling": r"/\sqrt{T}", "final": ""}[view]
    return "$" + numerator + suffix + "$"


def regret_comparison_axis_label(view: str) -> str:
    suffix = {"average": "/T", "sqrt_scaling": r"/\sqrt{T}"}[view]
    return "$R_T" + suffix + "$"


def finish_line_figure(figure, axes, legend_ncol=None) -> None:
    """Reserve a compact figure-level legend below the axes, at fixed physical width."""
    axes.grid(True)
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        figure.tight_layout(pad=0.7)
        return
    columns = legend_ncol or (1 if len(handles) <= 2 else 2)
    wrapped = [fill(label, width=72 if columns == 1 else 34) for label in labels]
    rows = (len(handles) + columns - 1) // columns
    extra_lines = sum(max(label.count("\n") for label in wrapped[i:i + columns])
                      for i in range(0, len(wrapped), columns))
    legend_height = 0.20 * (rows + extra_lines) + 0.12
    height = figure.get_figheight() + max(0, legend_height - 0.55)
    figure.set_size_inches(FIGURE_WIDTH, height)
    figure.tight_layout(rect=(0, legend_height / height, 1, 1), pad=0.7)
    figure.legend(handles, wrapped, loc="lower center", ncol=columns,
                  bbox_to_anchor=(0.5, 0.01), columnspacing=1.3, handlelength=2.6)


def heatmap(axes, values, *, vmax, label_format):
    """Vector cells and readable, integer action ticks, including large games."""
    from experiments.plots import HEATMAP_COLORMAP

    rows, columns = values.shape
    image = axes.pcolormesh(np.arange(columns + 1) - 0.5,
                           np.arange(rows + 1) - 0.5, values,
                           cmap=HEATMAP_COLORMAP, vmin=0, vmax=vmax,
                           rasterized=False)
    axes.set_aspect("equal")
    for count, setter in ((columns, axes.set_xticks), (rows, axes.set_yticks)):
        ticks = np.unique(np.r_[np.arange(0, count, max(1, (count + 9) // 10)), count - 1])
        setter(ticks)
    if values.size <= 100:
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            axes.text(column, row, label_format(value), ha="center", va="center",
                      fontsize=ANNOTATION_SIZE, color="white" if value > 0.55 * vmax else "#222222")
    return image
