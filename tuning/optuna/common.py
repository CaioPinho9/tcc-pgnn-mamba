from __future__ import annotations

from pathlib import Path
import math

import optuna
import pandas as pd


INVALID_TRIAL_REASON_ATTR = "invalid_reason"
CUDA_OOM_INVALID_REASON = "cuda_oom"


def include_value(candidates, value):
    if value is None:
        return list(candidates)
    out = list(candidates)
    if value not in out:
        out.append(value)
    try:
        return sorted(out)
    except TypeError:
        return out


def unique_presets(presets):
    seen = set()
    out = []
    for preset in presets:
        name = preset["name"]
        if name not in seen:
            out.append(preset)
            seen.add(name)
    return out


def ensure_epoch_column(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    if "epoch" not in log.columns:
        log["epoch"] = range(1, len(log) + 1)
    return log


def compute_history_objective(
        history: list[dict],
        metric: str,
        aggregate: str,
) -> tuple[float, int]:
    if not history:
        raise optuna.TrialPruned("O histórico de treinamento está vazio.")

    values = []
    epochs = []
    for row in history:
        if metric not in row:
            raise ValueError(f"A métrica {metric!r} não foi encontrada nas chaves da linha da época: {list(row.keys())}")
        value = float(row[metric])
        if not math.isfinite(value):
            continue
        values.append(value)
        epochs.append(int(row.get("epoch", len(epochs) + 1)))

    if not values:
        raise optuna.TrialPruned(f"Nenhum valor finito de {metric!r} foi produzido.")

    # Converta as métricas por época em um único valor comparável entre tentativas.
    if aggregate == "best":
        best_index = min(range(len(values)), key=lambda i: values[i])
        return float(values[best_index]), int(epochs[best_index])

    if aggregate == "last":
        return float(values[-1]), int(epochs[-1])

    if aggregate == "mean_last_10":
        window = values[-10:]
        return float(sum(window) / len(window)), int(epochs[-1])

    raise ValueError(f"Agregação desconhecida: {aggregate}")


def history_min(history: list[dict], metric: str) -> float | None:
    vals = [float(row[metric]) for row in history if metric in row and math.isfinite(float(row[metric]))]
    return min(vals) if vals else None


def history_last(history: list[dict], metric: str) -> float | None:
    vals = [float(row[metric]) for row in history if metric in row and math.isfinite(float(row[metric]))]
    return vals[-1] if vals else None


def stringify_for_csv(value):
    if isinstance(value, (list, tuple)):
        return str(value)
    return value


def append_optuna_summary(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    row_frame = pd.DataFrame([row])
    header = not summary_path.exists()
    row_frame.to_csv(summary_path, mode="a", header=header, index=False)


def counted_finished_trials(trials) -> list:
    return [
        trial
        for trial in trials
        if trial.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
    ]
