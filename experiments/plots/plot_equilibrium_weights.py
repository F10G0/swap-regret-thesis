from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.plots import save_figure_pair
from experiments.plots.style import publication_plot, HEATMAP_FIGURE_SIZE, heatmap
from metrics.equilibrium import equilibrium_profile_weights


@publication_plot
def plot_equilibrium_profile_weights(
    payoff_tensor,
    equilibrium: str,
    output_path: str | Path,
) -> None:
    """Render independently maximized equilibrium weights for a two-player game."""
    weights = equilibrium_profile_weights(
        payoff_tensor,
        equilibrium=equilibrium,
    )
    if weights.ndim != 2:
        raise ValueError(
            "equilibrium profile heatmaps require exactly two players"
        )

    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
    image = heatmap(axes, weights, vmax=1.0, label_format=lambda value: f"{value:.3f}")
    axes.set_xlabel("Player 1 action")
    axes.set_ylabel("Player 0 action")
    colorbar = figure.colorbar(image, ax=axes, label=f"Maximum {equilibrium.upper()} profile weight")
    colorbar.solids.set_rasterized(False)
    figure.tight_layout()
    save_figure_pair(figure, output_path)
    plt.close(figure)
