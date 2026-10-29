from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from config.config import OPENFWI_CONFIG


def _amplitude_clip(data, percentile):
    clip = np.percentile(np.abs(data), percentile)
    return clip if clip != 0 else 1.0


def plot_seismic_comparison(
    predicted_seismic_data,
    target_seismic_data,
    time_spacing_s=OPENFWI_CONFIG["time_spacing_s"],
    receiver_spacing_m=OPENFWI_CONFIG["receiver_line_spacing_m"],
    predicted_title="Dados sísmicos preditos",
    target_title="Dados sísmicos alvo",
    difference_title="Diferença",
    clip_percentile=99.0,
    difference_clip_percentile=99.0,
    figure_size=(18, 6),
    save_path=None,
    show=True,
    receiver_axis_label="Posição do receptor (m)",
):
    predicted = np.asarray(predicted_seismic_data, dtype=float)
    target = np.asarray(target_seismic_data, dtype=float)

    if predicted.ndim != 2 or target.ndim != 2:
        raise ValueError("As duas entradas sísmicas devem ter a forma (time_steps, receiver_count).")

    if predicted.shape != target.shape:
        raise ValueError(
            f"As entradas sísmicas devem ter a mesma forma. Obtidas {predicted.shape} e {target.shape}."
        )

    difference = predicted - target

    time_steps, receiver_count = predicted.shape
    extent = [
        0,
        (receiver_count - 1) * receiver_spacing_m,
        0,
        time_steps * time_spacing_s,
    ]

    shared_clip = _amplitude_clip(np.concatenate([predicted.ravel(), target.ravel()]), clip_percentile)
    diff_clip = _amplitude_clip(difference, difference_clip_percentile)

    fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)

    plots = [
        (predicted, predicted_title, "gray", shared_clip, "Amplitude"),
        (target, target_title, "gray", shared_clip, "Amplitude"),
        (difference, difference_title, "seismic", diff_clip, "Amplitude da diferença"),
    ]

    for ax, (data, title, cmap, clip, colorbar_label) in zip(axes, plots):
        image = ax.imshow(
            data,
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=-clip,
            vmax=clip,
        )
        ax.set(xlabel=receiver_axis_label, ylabel="Tempo (s)")
        fig.colorbar(image, ax=ax, label=colorbar_label)

    mae = np.mean(np.abs(difference))
    mse = np.mean(difference**2)
    max_abs_diff = np.max(np.abs(difference))

    metrics = f"MSE: {mse:.6e} | MAE: {mae:.6e} | Diferença absoluta máxima: {max_abs_diff:.6e}"

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return fig, axes

    plt.close(fig)
    print(metrics)
    return None
