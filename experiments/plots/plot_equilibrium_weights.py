from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.plots import HEATMAP_COLORMAP
from metrics.equilibrium import equilibrium_profile_weights


EQUILIBRIUM_TITLES = {
    "ce": "Maximum CE Profile Weight",
    "cce": "Maximum CCE Profile Weight",
}


def plot_equilibrium_profile_weights(
    payoff_tensor,
    equilibrium: str,
    output_path: str | Path,
    game_name: str | None = None,
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
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(
        weights,
        cmap=HEATMAP_COLORMAP,
        vmin=0.0,
        vmax=1.0,
    )
    axes.set_xlabel("Player 1 action")
    axes.set_ylabel("Player 0 action")
    title = EQUILIBRIUM_TITLES[equilibrium]
    axes.set_title(f"{game_name}: {title}" if game_name else title)
    axes.set_xticks(range(weights.shape[1]))
    axes.set_yticks(range(weights.shape[0]))

    if weights.size <= 100:
        for action_0, action_1 in np.ndindex(weights.shape):
            axes.text(
                action_1,
                action_0,
                f"{weights[action_0, action_1]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    figure.colorbar(image, ax=axes, label="Maximum profile weight")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
