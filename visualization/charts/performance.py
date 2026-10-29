from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from config.config import MODEL_NAMES
from .metrics import plot_metric_comparison


def plot_scenario_runtime_charts(
    runtime_table: pd.DataFrame,
    *,
    performance_definitions: Sequence[dict],
    output_dir: str | Path,
) -> list[Path]:
    """Plota os custos por base e geometria, com ruído e épocas reunidos."""
    output_dir = Path(output_dir)
    saved: list[Path] = []
    required = {"dataset_name", "model_name", "model_label", "geometry_description"}
    if runtime_table.empty or not required.issubset(runtime_table.columns):
        return saved

    for dataset_name, group in runtime_table.groupby(
        "dataset_name", sort=False, dropna=False
    ):
        slug = re.sub(r"[^a-z0-9]+", "_", str(dataset_name).casefold()).strip("_")
        for definition in performance_definitions:
            key = str(definition["key"])
            mean_column = f"evaluation_{key}_mean"
            std_column = f"evaluation_{key}_std"
            if mean_column not in group:
                continue
            selected_columns = ["geometry_description", "model_label", mean_column]
            if std_column in group:
                selected_columns.append(std_column)
            chart_values = group[selected_columns].rename(
                columns={mean_column: "mean", std_column: "std"}
            )
            labels_by_model = (
                group[["model_name", "model_label"]]
                .drop_duplicates("model_name")
                .set_index("model_name")["model_label"]
                .astype(str)
                .to_dict()
            )
            path = output_dir / f"{slug}_{key}_by_geometry.png"
            plot_metric_comparison(
                chart_values,
                x="geometry_description",
                y="mean",
                hue="model_label",
                yerr="std" if "std" in chart_values else None,
                title=f"{definition['label']} por geometria - {dataset_name}",
                ylabel=(
                    "Média das medianas (ms/amostra)"
                    if key == "inference_time_ms"
                    else "Memória GPU (MB)"
                ),
                save_path=path,
                x_order=list(dict.fromkeys(group["geometry_description"].astype(str))),
                hue_order=[
                    labels_by_model[name]
                    for name in MODEL_NAMES
                    if name in labels_by_model
                ],
                tick_digits=int(definition.get("digits", 2)),
                zero_floor=True,
            )
            if path.is_file():
                saved.append(path)
    return saved
