from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import _display_chart_label


def plot_sample_comparison(
    sample_id: int,
    true_velocity,
    predictions: dict[str, np.ndarray],
    *,
    vmin: float,
    vmax: float,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(predictions)
    cols = 1 + len(names)
    fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 10), constrained_layout=True)

    true_arr = np.squeeze(np.asarray(true_velocity))
    velocity_images = [("Gabarito", true_arr), *[(name, np.squeeze(predictions[name])) for name in names]]

    for col, (name, data) in enumerate(velocity_images):
        image = axes[0, col].imshow(data, cmap="jet", vmin=vmin, vmax=vmax)
        display_name = _display_chart_label(name)
        axes[0, col].set_title(display_name)
        axes[0, col].axis("off")
    fig.colorbar(image, ax=axes[0, :], fraction=0.025)

    for col, name in enumerate([""] + names):
        axes[1, col].axis("off")
        axes[2, col].axis("off")
        if not name:
            axes[1, col].set_title("Erro absoluto")
            axes[2, col].set_title("Erro com sinal")
            continue
        display_name = _display_chart_label(name)
        error = np.squeeze(predictions[name]) - true_arr
        abs_image = axes[1, col].imshow(np.abs(error), cmap="magma")
        signed_clip = max(float(np.max(np.abs(error))), 1e-12)
        signed_image = axes[2, col].imshow(error, cmap="seismic", vmin=-signed_clip, vmax=signed_clip)
        axes[1, col].set_title(f"Erro absoluto de {display_name}")
        axes[2, col].set_title(f"Erro com sinal de {display_name}")

    if names:
        fig.colorbar(abs_image, ax=axes[1, 1:], fraction=0.025)
        fig.colorbar(signed_image, ax=axes[2, 1:], fraction=0.025)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
