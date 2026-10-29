from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_losses(
    values: pd.DataFrame,
    *,
    model_label: str,
    title: str,
    save_path: str | Path,
) -> None:
    """Salva as curvas médias de perda dos dados e do modelo para um modelo."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"dataset_name", "epoch", "data_loss", "model_loss"}
    if values.empty or not required.issubset(values.columns):
        return

    values = values.copy()
    for column in ("epoch", "data_loss", "model_loss"):
        if column in values.columns:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=["dataset_name", "epoch", "data_loss", "model_loss"])
    if values.empty:
        return

    datasets = list(values["dataset_name"].astype(str).drop_duplicates())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(datasets))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

    for ax, metric, metric_title in (
        (axes[0], "data_loss", "Perda dos dados"),
        (axes[1], "model_loss", "Perda do modelo"),
    ):
        positive_values = values[metric].gt(0).all()
        if positive_values:
            ax.set_yscale("log")

        for color, dataset_name in zip(colors, datasets):
            curve = values[values["dataset_name"].astype(str) == dataset_name].sort_values("epoch")
            if curve.empty:
                continue
            ax.plot(
                curve["epoch"],
                curve[metric],
                color=color,
                linewidth=1.8,
                label=dataset_name,
            )

        ax.set_title(metric_title)
        ax.set_xlabel("Época")
        ax.set_ylabel("Perda")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(title or f"Perdas de treinamento - {model_label}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
