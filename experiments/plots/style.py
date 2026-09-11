"""Shared thesis-sized plotting defaults; no experiment or metric semantics."""

from functools import wraps
from textwrap import fill
from threading import RLock

import matplotlib as mpl
import numpy as np

from experiments.algorithm_labels import algorithm_label


FIGURE_WIDTH = 6.3
PUBLICATION_STYLE_VERSION = 1
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

# Color-vision-friendly colors plus redundant dash/marker identities.
ALGORITHM_STYLES = {
    "hedge": ("#0072B2", "-", "o"),
    "auer_exp3": ("#AA4499", "--", "s"),
    "exp3_ix": ("#332288", "-.", "^"),
    "bm": ("#D55E00", ":", "D"),
    "ito": ("#009E73", "--", "v"),
    "lce_ix": ("#882255", "-.", "P"),
    "regret_matching": ("#CC79A7", ":", "<"),
    "stationary_regret_matching": ("#E69F00", "-", "X"),
    "tsallis_inf": ("#444444", "--", ">"),
    "exp3": ("#666666", ":", "h"),
}


def publication_plot(function):
    """Apply typography during artist creation and saving, restoring rcParams."""
    @wraps(function)
    def styled(*args, **kwargs):
        with _STYLE_LOCK, mpl.rc_context(PUBLICATION_RC):
            return function(*args, **kwargs)
    return styled


def algorithm_style(name: str) -> dict:
    color, linestyle, marker = ALGORITHM_STYLES.get(name, ("#444444", "-", "o"))
    return dict(color=color, linestyle=linestyle, marker=marker, markevery=0.12)


def profile_label(profile) -> str:
    names = tuple(profile)
    if names and len(set(names)) == 1:
        names = names[:1]
    return " vs ".join(algorithm_label(name) for name in names)


def curve_labels(rows: list[dict]) -> list[str]:
    """Only distinguish metadata when the same profile appears more than once."""
    labels = [profile_label(row["algorithm"].split("_vs_")) for row in rows]
    fields = (
        ("feedback_mode", ""), ("horizon", "T="), ("seed", "seed "),
        ("base_learner_seed", "seed "), ("base_environment_seed", "env seed "),
        ("stationary_method", "solver "), ("replicate_count", "n="),
        ("implementation_version", "v"), ("runtime_fingerprint", "runtime "),
    )
    result = []
    for row, label in zip(rows, labels):
        peers = [other for other, other_label in zip(rows, labels) if other_label == label]
        details = [prefix + str(row.get(key, "")).replace("full_information", "full info")
                   for key, prefix in fields
                   if len({str(peer.get(key, "")) for peer in peers}) > 1]
        result.append(" · ".join([label, *details]))
    return result


def regret_axis_label(kind: str, view: str = "average") -> str:
    numerator = rf"R_T^{{\mathrm{{{kind}}}}}"
    suffix = {"average": "/T", "sqrt_scaling": r"/\sqrt{T}", "final": ""}[view]
    return "$" + numerator + suffix + "$"


def finish_line_figure(figure, axes) -> None:
    """Reserve a compact one/two-column legend below, at fixed physical width."""
    axes.grid(True)
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        figure.tight_layout(pad=0.7)
        return
    columns = 1 if len(handles) <= 2 else 2
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
