"""Tabelas de avaliação de cenários para apresentação no notebook."""

from __future__ import annotations

import math

import pandas as pd
from IPython.display import HTML, Markdown, display

from network.evaluation import model_label, scenario_presentation_table


GEOMETRY_LABELS = {
    "surface": "superfície",
    "surface_wells": "superfície + poços",
}


def _metric_column_label(metric: str, metric_config: dict) -> str:
    definition = metric_config[metric]
    direction = "↑" if definition["direction"] == "higher" else "↓"
    return f"{definition['label']} {direction}"


def _metric_mean_std(row: dict, metric: str, metric_config: dict) -> str:
    mean_value = pd.to_numeric(
        pd.Series([row.get(f"evaluation_{metric}_mean")]),
        errors="coerce",
    ).iloc[0]
    std_value = pd.to_numeric(
        pd.Series([row.get(f"evaluation_{metric}_std")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(mean_value):
        return ""
    digits = metric_config[metric]["digits"]
    mean_text = f"{mean_value:.{digits}f}".replace(".", ",")
    if pd.isna(std_value):
        return mean_text
    std_text = f"{std_value:.{digits}f}".replace(".", ",")
    return f"{mean_text} ± {std_text}"


def _scenario_key(row: dict):
    try:
        return (
            str(row["model_name"]),
            float(row["noise_std_ratio"]),
            int(row["epochs"]),
            str(row["receiver_geometry"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _build_scenario_table(
    dataset_name: str,
    factor_table: pd.DataFrame,
    presentation_config: dict,
    receiver_geometries: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    metric_config = {item["key"]: item for item in presentation_config["metrics"]}
    model_order = presentation_config["model_order"]
    measured_table = scenario_presentation_table(factor_table)
    measured_rows = {
        key: row
        for row in measured_table.to_dict(orient="records")
        if (key := _scenario_key(row)) is not None
    }
    noise_values = sorted(
        pd.to_numeric(factor_table["noise_std_ratio"], errors="coerce")
        .dropna()
        .unique()
    )
    epoch_values = sorted(
        pd.to_numeric(factor_table["epochs"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    rows = []
    scenario_end_rows = set()
    best_cells = {
        _metric_column_label(metric, metric_config): []
        for metric in metric_config
    }

    for noise_ratio in noise_values:
        for epochs in epoch_values:
            for geometry in receiver_geometries[dataset_name]:
                for model_name in model_order:
                    evaluated = measured_rows.get(
                        (model_name, float(noise_ratio), int(epochs), str(geometry)),
                        {},
                    )
                    noise = (
                        "Sem ruído"
                        if noise_ratio == 0
                        else f"Ruído {noise_ratio:g}".replace(".", ",")
                    )
                    rows.append({
                        "Conjunto de dados": dataset_name,
                        "Modelo": evaluated.get("model_label", model_label(model_name)),
                        "Ruído": noise,
                        "Geometria": GEOMETRY_LABELS.get(geometry, geometry),
                        **{
                            _metric_column_label(metric, metric_config): _metric_mean_std(
                                evaluated,
                                metric,
                                metric_config,
                            )
                            for metric in metric_config
                        },
                    })
                    for metric in metric_config:
                        best = str(evaluated.get(f"evaluation_{metric}_best", "")).strip()
                        best_cells[_metric_column_label(metric, metric_config)].append(
                            best in {"↑", "↓"}
                        )
                if rows:
                    scenario_end_rows.add(len(rows) - 1)

    table = pd.DataFrame(rows)
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    for column, mask in best_cells.items():
        styles.loc[mask, column] = "font-weight: bold"
    for row_index in scenario_end_rows:
        for column in styles.columns:
            style = styles.at[row_index, column]
            separator = "border-bottom: 5px double #CBD5E1"
            styles.at[row_index, column] = f"{style}; {separator}" if style else separator
    return table, styles, scenario_end_rows


def display_scenario_tables(
    scenario_results: dict,
    dataset_names: tuple[str, ...],
    presentation_config: dict,
    receiver_geometries: dict,
) -> None:
    """Exibe a tabela completa de cenários de cada conjunto de dados."""
    for dataset_name in dataset_names:
        table, styles, _ = _build_scenario_table(
            dataset_name,
            scenario_results[dataset_name].factor_table,
            presentation_config,
            receiver_geometries,
        )
        if table.empty:
            display(Markdown(
                f"### {dataset_name}\n\nNão há cenários para exibir na versão "
                f"`{presentation_config['version']}`. Confira se há checkpoints dessa versão."
            ))
            continue

        display(Markdown(
            f"### {dataset_name}\n\nMédia ± desvio padrão entre amostras. "
            "O melhor resultado entre os modelos em cada cenário está em negrito. "
            "Linhas duplas separam cenários; células vazias indicam cenários sem avaliações concluídas."
        ))
        display(table.style.apply(lambda _: styles, axis=None).hide(axis="index").set_table_styles([
            {"selector": "th", "props": [("background-color", "#111827"), ("color", "white"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "5px 8px"), ("border-bottom", "1px solid #CBD5E1")]},
        ]))


def _sample_win_percentages(
    sample_metrics: pd.DataFrame,
    metric_config: dict,
    model_order: list[str],
) -> pd.DataFrame:
    comparison_columns = [
        "noise_std_ratio",
        "epochs_target",
        "receiver_geometry",
        "sample_id",
    ]
    if "dataset_name" in sample_metrics.columns:
        comparison_columns.insert(0, "dataset_name")
    comparison_groups = sample_metrics.groupby(
        comparison_columns,
        sort=False,
        dropna=False,
    )
    wins = {
        metric: {model_name: 0.0 for model_name in model_order}
        for metric in metric_config
    }
    eligible_counts = {metric: 0 for metric in metric_config}

    for metric, definition in metric_config.items():
        for _, group in comparison_groups:
            model_rows = group.set_index("model_name")
            if model_rows.index.has_duplicates or not set(model_order).issubset(model_rows.index):
                continue
            values = pd.to_numeric(
                model_rows[metric].reindex(model_order),
                errors="coerce",
            )
            if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
                continue
            best_value = values.max() if definition["direction"] == "higher" else values.min()
            winners = values.index[values.eq(best_value)].tolist()
            if not winners:
                continue
            credit = 1.0 / len(winners)
            for model_name in winners:
                wins[metric][model_name] += credit
            eligible_counts[metric] += 1

    labels = sample_metrics.drop_duplicates("model_name").set_index("model_name")["model_label"].to_dict()
    rows = []
    for model_name in model_order:
        row = {"Modelo": labels.get(model_name, model_label(model_name))}
        for metric in metric_config:
            denominator = eligible_counts[metric]
            column = f"{_metric_column_label(metric, metric_config)} (n={denominator})"
            row[column] = (
                f"{100.0 * wins[metric][model_name] / denominator:.1f}%".replace(".", ",")
                if denominator else "N/A"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _scenario_sample_win_percentages(
    sample_metrics: pd.DataFrame,
    metric_config: dict,
    model_order: list[str],
) -> pd.DataFrame:
    required_columns = {
        "dataset_name",
        "epochs_target",
        "noise_std_ratio",
        "receiver_geometry",
        "sample_id",
        "model_name",
        "model_label",
        *metric_config,
    }
    missing_columns = required_columns.difference(sample_metrics.columns)
    if missing_columns:
        raise ValueError(
            "A tabela de métricas por amostra não contém as colunas: "
            + ", ".join(sorted(missing_columns))
        )

    frequencies = {}
    scenario_keys = []
    scenario_groups = sample_metrics.groupby(
        ["noise_std_ratio", "receiver_geometry"],
        sort=True,
        dropna=False,
    )
    for scenario_key, scenario_rows in scenario_groups:
        noise_ratio, geometry = scenario_key
        noise_text = "Sem ruído" if float(noise_ratio) == 0.0 else (
            f"Ruído {float(noise_ratio):g}".replace(".", ",")
        )
        label = f"{noise_text} | {GEOMETRY_LABELS.get(str(geometry), str(geometry))}"
        scenario_keys.append((label, scenario_key))
        scenario_wins = {
            metric: {model_name: 0.0 for model_name in model_order}
            for metric in metric_config
        }
        eligible_counts = {metric: 0 for metric in metric_config}

        comparison_groups = scenario_rows.groupby(
            ["dataset_name", "epochs_target", "sample_id"],
            sort=False,
            dropna=False,
        )
        for _, sample_rows in comparison_groups:
            model_rows = sample_rows.set_index("model_name")
            if model_rows.index.has_duplicates or not set(model_order).issubset(model_rows.index):
                continue
            for metric, definition in metric_config.items():
                values = pd.to_numeric(
                    model_rows[metric].reindex(model_order),
                    errors="coerce",
                )
                if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
                    continue
                best_value = values.max() if definition["direction"] == "higher" else values.min()
                winners = values.index[values.eq(best_value)].tolist()
                if not winners:
                    continue
                credit = 1.0 / len(winners)
                for model_name in winners:
                    scenario_wins[metric][model_name] += credit
                eligible_counts[metric] += 1
        frequencies[scenario_key] = (scenario_wins, eligible_counts)

    columns = ["Modelo"] + [label for label, _ in scenario_keys]
    labels = sample_metrics.drop_duplicates("model_name").set_index("model_name")["model_label"].to_dict()
    rows = []
    for model_name in model_order:
        row = {"Modelo": labels.get(model_name, model_label(model_name))}
        for label, scenario_key in scenario_keys:
            scenario_wins, eligible_counts = frequencies[scenario_key]
            percentages = []
            for metric in ("mse", "mae", "ssim"):
                if metric not in metric_config:
                    continue
                denominator = eligible_counts[metric]
                credit = scenario_wins[metric][model_name]
                percentages.append(
                    f"{100.0 * credit / denominator:.1f}%".replace(".", ",")
                    if denominator else "N/A"
                )
            row[label] = f"({'; '.join(percentages)})"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _display_comparison_table(table: pd.DataFrame, *, centered: bool = False) -> None:
    alignment = "center" if centered else "left"
    styles = f"""<style>
    table.comparative-analysis-table {{ border-collapse: collapse; margin: 0; }}
    table.comparative-analysis-table th {{ background-color: #111827; color: white; text-align: {alignment}; padding: 5px 8px; }}
    table.comparative-analysis-table td {{ padding: 5px 8px; border-bottom: 1px solid #CBD5E1; text-align: {alignment}; }}
    table.comparative-analysis-table tbody tr:nth-child(even) {{ background-color: #F8FAFC; }}
    </style>"""
    display(HTML(styles + table.to_html(index=False, classes="comparative-analysis-table", border=0)))


def display_scenario_comparative_analysis(
    scenario_results: dict,
    dataset_names: tuple[str, ...],
    presentation_config: dict,
) -> None:
    """Exibe a análise comparativa dos modelos por cenário e por amostra."""
    metric_config = {item["key"]: item for item in presentation_config["metrics"]}
    model_order = presentation_config["model_order"]
    all_sample_metrics = []

    for dataset_name in dataset_names:
        result = scenario_results[dataset_name]
        sample_metrics = pd.DataFrame(result.output.rows)
        if not sample_metrics.empty and "status" in sample_metrics.columns:
            sample_metrics = sample_metrics.loc[sample_metrics["status"].eq("ok")].copy()
        if sample_metrics.empty:
            display(Markdown(
                f"### {dataset_name}\n\nNão há métricas por amostra válidas para comparar."
            ))
            continue
        sample_metrics["dataset_name"] = dataset_name
        all_sample_metrics.append(sample_metrics)

        sample_percentages = _sample_win_percentages(
            sample_metrics,
            metric_config,
            model_order,
        )
        display(Markdown(f"### {dataset_name}"))
        _display_comparison_table(sample_percentages)

    if not all_sample_metrics:
        return

    total_sample_metrics = pd.concat(all_sample_metrics, ignore_index=True)
    total_sample_percentages = _sample_win_percentages(
        total_sample_metrics,
        metric_config,
        model_order,
    )
    display(Markdown("### Total — todos os conjuntos de dados e amostras"))
    _display_comparison_table(total_sample_percentages)

    sample_comparison_percentages = _scenario_sample_win_percentages(
        total_sample_metrics,
        metric_config,
        model_order,
    )
    display(Markdown(
        "### Análise comparativa por cenário e métrica — todos os conjuntos de dados"
    ))
    display(Markdown(
        "Cada coluna reúne um nível de ruído e uma geometria. Cada comparação válida corresponde a uma amostra do mesmo conjunto de dados e época; "
        "cada célula mostra percentuais de vitória para MSE, MAE e SSIM, nessa ordem e separados por ponto e vírgula. Empates dividem o crédito."
    ))
    _display_comparison_table(sample_comparison_percentages, centered=True)
