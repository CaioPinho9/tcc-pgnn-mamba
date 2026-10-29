from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

from config.config import MODEL_NAMES
from .metrics import _display_chart_label


def plot_velocity_profiles(
    true_velocity: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    grid_spacing_m: float,
    title: str,
    save_path: str | Path,
    profile_positions: Sequence[float] = (0.25, 0.5, 0.75),
    model_order: Sequence[str] | None = None,
) -> None:
    """Plota perfis verticais de velocidade em frações da extensão horizontal do modelo.

        Os arrays contêm velocidades físicas em m/s e têm formato ``(profundidade, largura)``.
        As posições vão de 0 (borda esquerda) a 1 (borda direita) e selecionam a coluna
        de grade mais próxima. A profundidade e a distância horizontal começam na origem
        do modelo.
    """
    if not predictions:
        return
    true_arr = np.asarray(true_velocity)
    if true_arr.ndim != 2 or not all(true_arr.shape):
        raise ValueError("Velocity profiles require a non-empty (depth, width) array")
    if not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be finite and positive")
    positions = np.asarray(profile_positions, dtype=float)
    if (
        positions.ndim != 1 or positions.size == 0
        or not np.all(np.isfinite(positions))
        or np.any((positions < 0) | (positions > 1))
    ):
        raise ValueError("profile_positions must contain fractions between 0 and 1")
    arrays = {name: np.asarray(values) for name, values in predictions.items()}
    for name, values in arrays.items():
        if values.shape != true_arr.shape:
            raise ValueError(f"Prediction {name!r} must match the ground-truth shape")

    names = list(dict.fromkeys(name for name in (model_order or ()) if name in arrays))
    names.extend(name for name in arrays if name not in names)
    columns = list(dict.fromkeys(np.rint(positions * (true_arr.shape[1] - 1)).astype(int)))
    depth_m = np.arange(true_arr.shape[0]) * grid_spacing_m
    colors = plt.get_cmap("tab10").colors
    line_styles = ("--", "-.", ":")
    fig, axes = plt.subplots(
        1, len(columns), figsize=(3.6 * len(columns), 6.5),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax, column in zip(axes[0], columns):
        ax.plot(true_arr[:, column], depth_m, color="black", linewidth=1.8, label="Gabarito")
        for index, name in enumerate(names):
            ax.plot(
                arrays[name][:, column], depth_m,
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)],
                linewidth=1.3, label=_display_chart_label(name),
            )
        ax.set_title(f"x = {column * grid_spacing_m:g} m")
        ax.set_xlabel(r"$V_p$ (m/s)")
        ax.grid(True, color="0.75", linewidth=0.6)
        ax.set_axisbelow(True)
    # Shared axes need to be inverted only once.
    axes[0, 0].set_ylim(max(float(depth_m[-1]), grid_spacing_m), 0)
    axes[0, 0].set_ylabel("Profundidade (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_columns = min(3, len(labels)) if len(columns) > 1 else 1
    legend_rows = int(np.ceil(len(labels) / legend_columns))
    legend_height = 0.04 * legend_rows + 0.02
    fig.legend(handles, labels, loc="lower center", ncol=legend_columns, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, legend_height, 1, 0.95))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_scenario_comparison(
    sample_id: int,
    true_velocity,
    predictions: dict[str, np.ndarray],
    *,
    vmin: float,
    vmax: float,
    save_path: str | Path,
) -> None:
    """Compara o mapa alvo e as predições de velocidade de cenários de treinamento."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    true_arr = np.squeeze(np.asarray(true_velocity))
    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, prediction in predictions.items():
        parts = [part.strip() for part in name.split("|", maxsplit=1)]
        model_name = parts[0]
        scenario_name = parts[1] if len(parts) > 1 else ""
        model_predictions.setdefault(model_name, {})[scenario_name] = prediction

    scenario_names = list(dict.fromkeys(
        scenario_name
        for scenarios in model_predictions.values()
        for scenario_name in scenarios
    ))
    model_names = [name for name in MODEL_NAMES if name in model_predictions]
    model_names.extend(name for name in model_predictions if name not in model_names)
    if not model_names:
        return
    column_count = len(model_names) + 1
    row_count = len(scenario_names)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(2.4 * column_count, 2.15 * row_count),
        squeeze=False,
        constrained_layout=False,
    )
    axes[0, 0].set_title("Referência", fontsize=9)
    for column, model_name in enumerate(model_names, start=1):
        axes[0, column].set_title(model_name, fontsize=9)

    for row, scenario_name in enumerate(scenario_names):
        reference_axis = axes[row, 0]
        reference_axis.imshow(true_arr, cmap="jet", vmin=vmin, vmax=vmax, interpolation="nearest")
        reference_axis.axis("off")
        for column, model_name in enumerate(model_names, start=1):
            axis = axes[row, column]
            prediction = model_predictions.get(model_name, {}).get(scenario_name)
            if prediction is None:
                axis.axis("off")
                continue
            axis.imshow(
                np.squeeze(np.asarray(prediction)),
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.axis("off")

    top = 0.95 if row_count == 1 else 0.97
    fig.subplots_adjust(left=0.20, right=0.94, top=top, bottom=0.03, wspace=0.03, hspace=0.10)
    for row, scenario_name in enumerate(scenario_names):
        position = axes[row, 0].get_position()
        display_name = _display_chart_label(scenario_name)
        wrapped_name = "\n".join(part.strip() for part in display_name.split("|"))
        fig.text(0.195, position.y0 + position.height / 2, wrapped_name, ha="right", va="center", fontsize=7)
    colorbar_axis = fig.add_axes((0.955, 0.15, 0.012, 0.70))
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="jet"),
        cax=colorbar_axis,
        label="Velocidade (m/s)",
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current > previous + 1:
            ranges.append((start, previous + 1))
            start = current
        previous = current
    ranges.append((start, previous + 1))
    return ranges


def _velocity_panel_ranges(image: Image.Image) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pixels = np.asarray(image.convert("RGB"))
    color_spread = pixels.max(axis=2) - pixels.min(axis=2)
    colored = color_spread > 60
    row_ranges = [
        (start, stop)
        for start, stop in _contiguous_ranges(np.flatnonzero(colored.sum(axis=1) >= 100))
        if stop - start >= 80
    ]
    column_ranges = [
        (start, stop)
        for start, stop in _contiguous_ranges(np.flatnonzero(colored.sum(axis=0) >= 100))
        if stop - start >= 80
    ]
    return row_ranges, column_ranges


def reflow_cached_velocity_scenario_figure(
    image_path: str | Path,
    *,
    sample_id: int,
    scenario_factors: pd.DataFrame,
    model_labels: dict[str, str],
    vmin: float,
    vmax: float,
) -> bool:
    """Re-layout a cached velocity map into the common scenario-row/model-column grid."""
    image_path = Path(image_path)
    if not image_path.is_file() or scenario_factors.empty or not model_labels:
        return False

    factor_columns = ("noise_std_ratio", "epochs", "receiver_geometry")
    if not set(factor_columns).issubset(scenario_factors.columns):
        return False
    factors = scenario_factors.loc[:, factor_columns].drop_duplicates().copy()
    factors["noise_std_ratio"] = pd.to_numeric(factors["noise_std_ratio"], errors="coerce")
    factors["epochs"] = pd.to_numeric(factors["epochs"], errors="coerce")
    factors = factors.dropna(subset=list(factor_columns)).sort_values(
        list(factor_columns), kind="stable"
    ).reset_index(drop=True)
    model_names = [name for name in MODEL_NAMES if name in model_labels]
    model_names.extend(name for name in model_labels if name not in model_names)
    if factors.empty or not model_names:
        return False

    with Image.open(image_path) as source:
        source = source.convert("RGB")
        row_ranges, column_ranges = _velocity_panel_ranges(source)
        source_pixels = np.asarray(source)

    scenario_count = len(factors)
    is_current_layout = (
        len(row_ranges) == scenario_count
        and len(column_ranges) == len(model_names) + 1
    )
    is_legacy_layout = (
        len(row_ranges) == len(model_names)
        and len(column_ranges) >= 2
    )
    if not is_current_layout and not is_legacy_layout:
        return False

    def panel(row: int, column: int) -> np.ndarray:
        y0, y1 = row_ranges[row]
        x0, x1 = column_ranges[column]
        return source_pixels[y0:y1, x0:x1]

    if is_legacy_layout and not is_current_layout:
        available_scenarios = len(column_ranges) - 1
        if available_scenarios != scenario_count:
            epoch_groups = factors.groupby("epochs", sort=True)
            matching_epoch_groups = [
                group.reset_index(drop=True)
                for _, group in epoch_groups
                if len(group) == available_scenarios
            ]
            if len(matching_epoch_groups) != 1:
                return False
            factors = matching_epoch_groups[0]
            scenario_count = len(factors)

        legacy_model_order = (
            "mamba",
            "pgnn_mardan_2024",
            "pgann_mardan_2024",
            "unet",
            "unetmamba",
        )
        if len(row_ranges) != len(legacy_model_order) or not set(model_names).issubset(legacy_model_order):
            return False
        source_row_for_model = {
            model_name: row
            for row, model_name in enumerate(legacy_model_order)
        }
        panels = [
            [
                panel(source_row_for_model[model_names[0]], 0),
                *[
                    panel(source_row_for_model[model_name], scenario_index + 1)
                    for model_name in model_names
                ],
            ]
            for scenario_index in range(scenario_count)
        ]
        legacy_dir = image_path.parent / "legacy_layout"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_copy = legacy_dir / image_path.name
        if not legacy_copy.exists():
            shutil.copy2(image_path, legacy_copy)
    else:
        panels = [
            [panel(scenario_index, column) for column in range(len(model_names) + 1)]
            for scenario_index in range(scenario_count)
        ]

    row_labels = []
    for row in factors.to_dict(orient="records"):
        noise_ratio = float(row["noise_std_ratio"])
        noise_label = "sem ruído" if noise_ratio == 0 else f"ruído {noise_ratio:g}".replace(".", ",")
        geometry = str(row["receiver_geometry"])
        geometry_label = {
            "surface": "superfície",
            "surface_wells": "superfície + poços",
        }.get(geometry, geometry.replace("_", " "))
        row_labels.append(_display_chart_label(
            " | ".join((noise_label, f"{int(row['epochs'])} épocas", geometry_label))
        ).replace(" | ", "\n"))

    column_count = len(model_names) + 1
    row_count = len(row_labels)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(2.4 * column_count, 2.15 * row_count),
        squeeze=False,
        constrained_layout=False,
    )
    axes[0, 0].set_title("Referência", fontsize=9)
    for column, model_name in enumerate(model_names, start=1):
        axes[0, column].set_title(model_labels[model_name], fontsize=9)
    for row_index, row_panels in enumerate(panels):
        for column, panel_pixels in enumerate(row_panels):
            axes[row_index, column].imshow(panel_pixels, interpolation="nearest")
            axes[row_index, column].axis("off")

    top = 0.95 if row_count == 1 else 0.97
    fig.subplots_adjust(left=0.20, right=0.94, top=top, bottom=0.03, wspace=0.03, hspace=0.10)
    for row_index, row_label in enumerate(row_labels):
        position = axes[row_index, 0].get_position()
        fig.text(0.195, position.y0 + position.height / 2, row_label, ha="right", va="center", fontsize=7)
    colorbar_axis = fig.add_axes((0.955, 0.15, 0.012, 0.70))
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="jet"),
        cax=colorbar_axis,
        label="Velocidade (m/s)",
    )
    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True
