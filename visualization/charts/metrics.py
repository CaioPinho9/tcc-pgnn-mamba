from __future__ import annotations

from pathlib import Path
import re
import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from config.config import MODEL_NAMES


def _display_chart_label(label: object) -> str:
    parts = [part.strip() for part in str(label).split("|")]
    parts = [
        part for part in parts
        if not re.fullmatch(r"500\s+épocas", part, flags=re.IGNORECASE)
    ]
    return " | ".join(parts)


def _wrapped_category_labels(labels: Sequence[str], *, width: int = 18) -> list[str]:
    wrapped = []
    for label in labels:
        display_label = _display_chart_label(label)
        parts = [part.strip() for part in display_label.split("|")]
        if len(parts) > 1:
            wrapped.append("\n".join(parts))
        else:
            wrapped.append(textwrap.fill(display_label, width=width))
    return wrapped


def _format_metric_tick(value: float, _position: int, digits: int = 2) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.{digits}f}k"
    return f"{value:.{digits}f}"


def plot_metric_comparison(
    values: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    yerr: str | None = None,
    title: str,
    ylabel: str,
    save_path: str | Path,
    x_order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
    tick_digits: int = 2,
    zero_floor: bool = False,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    values = values.copy()
    if values.empty or x not in values.columns or y not in values.columns:
        return

    values[y] = pd.to_numeric(values[y], errors="coerce")
    values = values.dropna(subset=[x, y])
    if yerr is not None and yerr in values.columns:
        values[yerr] = pd.to_numeric(values[yerr], errors="coerce").fillna(0.0).clip(lower=0.0)
    if hue is not None:
        values = values.dropna(subset=[hue])
    if values.empty:
        return

    if x_order is not None:
        present = set(values[x].astype(str))
        ordered = [str(value) for value in x_order if str(value) in present]
        ordered.extend(
            value for value in values[x].astype(str).drop_duplicates()
            if value not in ordered
        )
        values[x] = pd.Categorical(values[x].astype(str), categories=ordered, ordered=True)
        values = values.sort_values(x)

    category_labels = list(dict.fromkeys(values[x].astype(str)))
    if x_order is not None:
        category_labels = [str(label) for label in x_order if str(label) in category_labels]
    category_count = len(category_labels)
    series_labels = []
    if hue is not None:
        present = set(values[hue].astype(str))
        series_labels = (
            [str(label) for label in hue_order if str(label) in present]
            if hue_order is not None
            else sorted(present)
        )
    fig_width = max(8, 1.8 * category_count) + (2 if hue is not None else 0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    if hue is None:
        ordered_values = values.copy()
        ordered_values["_plot_category"] = pd.Categorical(
            ordered_values[x].astype(str),
            categories=category_labels,
            ordered=True,
        )
        ordered_values = ordered_values.sort_values("_plot_category")
        positions = np.arange(len(ordered_values))
        ax.bar(
            positions,
            ordered_values[y],
            yerr=ordered_values[yerr] if yerr is not None and yerr in ordered_values else None,
            capsize=3 if yerr is not None and yerr in ordered_values else 0,
        )
        ax.set_xticks(
            positions,
            labels=_wrapped_category_labels(ordered_values[x].astype(str)),
            fontsize=8,
        )
    else:
        pivot = values.pivot(index=x, columns=hue, values=y)
        pivot = pivot.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if pivot.empty:
            plt.close(fig)
            return
        pivot.index = pivot.index.astype(str)
        pivot.columns = pivot.columns.astype(str)
        pivot = pivot.reindex(index=category_labels, columns=series_labels)
        error_pivot = None
        if yerr is not None and yerr in values.columns:
            error_pivot = values.pivot(index=x, columns=hue, values=yerr)
            error_pivot.index = error_pivot.index.astype(str)
            error_pivot.columns = error_pivot.columns.astype(str)
            error_pivot = error_pivot.reindex(
                index=category_labels,
                columns=series_labels,
            ).fillna(0.0)
        positions = np.arange(category_count)
        bar_width = 0.8 / max(len(series_labels), 1)
        for series_index, series_label in enumerate(series_labels):
            offset = (series_index - (len(series_labels) - 1) / 2) * bar_width
            ax.bar(
                positions + offset,
                pivot[series_label],
                width=bar_width,
                yerr=error_pivot[series_label] if error_pivot is not None else None,
                capsize=3 if error_pivot is not None else 0,
                label=_display_chart_label(series_label),
            )
        ax.set_xticks(positions, labels=_wrapped_category_labels(category_labels), fontsize=8)
        ax.legend(title=None, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    if title:
        ax.set_title(title, fontsize=15)
    ax.set_ylabel("" if ylabel == "Média" else ylabel, fontsize=14)
    ax.set_xlabel("", fontsize=14)
    if zero_floor:
        ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position: _format_metric_tick(value, position, tick_digits)
        )
    )
    fig.subplots_adjust(
        left=0.18 if zero_floor else None,
        bottom=0.30,
        top=0.96 if not title else 0.86,
        right=0.80 if hue is not None else 0.97,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_scenario_metric_charts(
    grouped_tables: dict[str, pd.DataFrame],
    *,
    metric_definitions: Sequence[dict],
    output_dir: str | Path,
) -> dict[str, list[Path]]:
    """Plota médias por fator com uma série independente para cada modelo."""
    output_dir = Path(output_dir)
    group_labels = {
        "noise": "ruído",
        "geometry": "geometria",
        "model": "modelo em todos os cenários",
    }
    saved: dict[str, list[Path]] = {name: [] for name in group_labels}
    for group_name, table in grouped_tables.items():
        if table.empty or "category" not in table:
            continue
        for definition in metric_definitions:
            key = str(definition["key"])
            mean_column = f"evaluation_{key}_mean"
            if mean_column not in table:
                continue
            chart_values = table[
                ["category", "model_name", "model_label", mean_column]
            ].copy()
            chart_values = chart_values.rename(columns={mean_column: "mean"}).dropna(subset=["mean"])
            categories = list(dict.fromkeys(chart_values["category"].astype(str)))
            present_models = set(chart_values["model_name"].astype(str))
            model_names = [name for name in MODEL_NAMES if name in present_models]
            model_names.extend(
                name
                for name in chart_values["model_name"].astype(str).drop_duplicates()
                if name not in model_names
            )
            label_by_model = (
                chart_values.drop_duplicates("model_name")
                .set_index("model_name")["model_label"]
                .astype(str)
                .to_dict()
            )
            model_order = [label_by_model[name] for name in model_names]
            path = output_dir / f"{group_name}_{key}_mean.png"
            group_label = group_labels.get(group_name, group_name)
            title = (
                f"{definition['label']} média por modelo"
                if group_name == "model"
                else f"{definition['label']} média por {group_label}"
            )
            plot_metric_comparison(
                chart_values,
                x="category",
                y="mean",
                hue=None if group_name == "model" else "model_label",
                title=title,
                ylabel=str(definition["label"]),
                save_path=path,
                x_order=categories,
                hue_order=None if group_name == "model" else model_order,
                tick_digits=int(definition.get("digits", 2)),
            )
            if path.is_file():
                saved.setdefault(group_name, []).append(path)
    return saved
