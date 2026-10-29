from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.config import MODEL_NAMES
from .metrics import plot_metric_comparison


def plot_scenario_charts(
    scenario_table: pd.DataFrame,
    *,
    dataset_name: str,
    output_dir: str | Path,
    metric_definitions: Sequence[dict],
    performance_definitions: Sequence[dict],
) -> dict[str, list[Path]]:
    """Cria as figuras de qualidade e custo agrupadas pelos fatores do cenário."""
    output_dir = Path(output_dir)
    required = {
        "model_name",
        "model_label",
        "noise_std_ratio",
        "geometry_description",
    }
    result = {"metrics": [], "performance": []}
    if scenario_table.empty or not required.issubset(scenario_table.columns):
        return result

    values = scenario_table.copy()
    values["noise_std_ratio"] = pd.to_numeric(values["noise_std_ratio"], errors="coerce")
    values = values.dropna(subset=["noise_std_ratio"])
    if values.empty:
        return result

    def noise_label(ratio: float) -> str:
        if ratio == 0:
            return "sem ruído"
        return f"ruído {ratio:g}".replace(".", ",")

    values["scenario_label"] = values.apply(
        lambda row: (
            f"{noise_label(float(row['noise_std_ratio']))} | "
            f"{row['geometry_description']}"
        ),
        axis=1,
    )
    scenario_order = list(dict.fromkeys(values["scenario_label"].astype(str)))
    present_models = set(values["model_name"].astype(str))
    model_names = [name for name in MODEL_NAMES if name in present_models]
    model_names.extend(sorted(present_models.difference(model_names)))
    labels_by_model = (
        values[["model_name", "model_label"]]
        .drop_duplicates("model_name")
        .set_index("model_name")["model_label"]
        .astype(str)
        .to_dict()
    )
    model_order = [labels_by_model[name] for name in model_names if name in labels_by_model]

    for definition in metric_definitions:
        key = str(definition["key"])
        mean_column = f"evaluation_{key}_mean"
        if mean_column not in values.columns:
            continue
        chart_values = values[["scenario_label", "model_label", mean_column]].copy()
        chart_values = chart_values.rename(columns={mean_column: "mean"})
        path = output_dir / "figures" / "metrics" / "by_scenario" / f"{dataset_name}_{key}_by_scenario.png"
        plot_metric_comparison(
            chart_values,
            x="scenario_label",
            y="mean",
            hue="model_label",
            title="",
            ylabel=str(definition["label"]),
            save_path=path,
            x_order=scenario_order,
            hue_order=model_order,
            tick_digits=int(definition.get("digits", 2)),
        )
        if path.is_file():
            result["metrics"].append(path)

    # Save one chart per scenario and metric, but keep these files out of the
    # returned list so the evaluation notebook does not display them.
    individual_chart_dir = (
        output_dir / "figures" / "metrics" / "by_scenario" / "individual"
    )
    for scenario_label in scenario_order:
        selected = values.loc[values["scenario_label"].astype(str).eq(scenario_label)]
        ascii_label = (
            unicodedata.normalize("NFKD", scenario_label)
            .encode("ascii", "ignore")
            .decode()
        )
        scenario_slug = re.sub(r"[^a-z0-9]+", "_", ascii_label.lower()).strip("_")
        for definition in metric_definitions:
            key = str(definition["key"])
            mean_column = f"evaluation_{key}_mean"
            if mean_column not in selected.columns:
                continue
            chart_values = selected[["scenario_label", "model_label", mean_column]].copy()
            chart_values = chart_values.rename(columns={mean_column: "mean"})
            path = individual_chart_dir / (
                f"{dataset_name}_{scenario_slug}_{key}.png"
            )
            plot_metric_comparison(
                chart_values,
                x="scenario_label",
                y="mean",
                hue="model_label",
                title="",
                ylabel=str(definition["label"]),
                save_path=path,
                x_order=[scenario_label],
                hue_order=model_order,
                tick_digits=int(definition.get("digits", 2)),
            )

    for definition in performance_definitions:
        key = str(definition["key"])
        mean_column = f"evaluation_{key}_mean"
        std_column = f"evaluation_{key}_std"
        if mean_column not in values.columns:
            continue
        selected_columns = ["scenario_label", "model_label", mean_column]
        if std_column in values.columns:
            selected_columns.append(std_column)
        chart_values = values[selected_columns].copy()
        chart_values = chart_values.rename(columns={mean_column: "mean", std_column: "std"})
        path = output_dir / "figures" / "runtime" / "by_scenario" / f"{key}_by_scenario_all_scenarios.png"
        plot_metric_comparison(
            chart_values,
            x="scenario_label",
            y="mean",
            hue="model_label",
            yerr="std" if "std" in chart_values else None,
            title=f"{definition['label']} por cenário de treinamento - {dataset_name}",
            ylabel=(
                "Média das medianas (ms/amostra)"
                if key == "inference_time_ms"
                else "Pico de memória GPU (MB)"
                if key == "peak_gpu_memory_mb"
                else f"{definition['label']} ({definition['unit']})"
            ),
            save_path=path,
            x_order=scenario_order,
            hue_order=model_order,
            tick_digits=int(definition.get("digits", 2)),
        )
        if path.is_file():
            result["performance"].append(path)
    return result
