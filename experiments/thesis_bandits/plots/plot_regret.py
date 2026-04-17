import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def plot_single_cumulative_regret(
    cumulative_regret: np.ndarray,
    title: str = "Cumulative Regret",
    xlabel: str = "Round",
    ylabel: str = "Cumulative Regret",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot cumulative regret for a single run.

    Args:
        cumulative_regret: 1D array of cumulative regret values
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        save_path: optional path to save the figure
        show: whether to display the figure
    """
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_regret)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


def plot_average_cumulative_regret(
    avg_cumulative_regret: np.ndarray,
    title: str = "Average Cumulative Regret",
    xlabel: str = "Round",
    ylabel: str = "Average Cumulative Regret",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot average cumulative regret over multiple runs.

    Args:
        avg_cumulative_regret: 1D array of averaged cumulative regret
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        save_path: optional path to save figure
        show: whether to display the figure
    """
    plt.figure(figsize=(8, 5))
    plt.plot(avg_cumulative_regret)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


def plot_multiple_average_regrets(
    curves: dict[str, np.ndarray],
    title: str = "Algorithm Comparison",
    xlabel: str = "Round",
    ylabel: str = "Average Cumulative Regret",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot multiple average cumulative regret curves on one figure.

    Args:
        curves: dictionary mapping algorithm name -> 1D regret curve
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        save_path: optional path to save figure
        show: whether to display the figure
    """
    if len(curves) == 0:
        raise ValueError("curves must not be empty")

    plt.figure(figsize=(8, 5))

    for name, curve in curves.items():
        plt.plot(curve, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()
    