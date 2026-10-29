"""Medição pareada de memória GPU com blocos Mamba ativados e desativados."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config.config import resolve_device
from data.dataset import prepare_data, prepare_training_sample
from helpers.checkpointing import checkpoint_hyperparameters
from helpers.runtime import peak_gpu_memory_mb
from network.evaluation import (
    EvaluationJob,
    apply_evaluation_noise,
    build_run_config,
    load_checkpoint_low_memory,
    model_label,
)
from network.models.factory import build_model


MAMBA_MEMORY_MODELS = ("mamba", "unetmamba")
MAMBA_SWITCHES = (
    "use_mamba_encoder",
    "use_mamba_bottleneck",
    "use_mamba_decoder",
)
MEMORY_TABLE_COLUMNS = (
    "Conjunto de dados",
    "Modelo",
    "Comparações pareadas (n)",
    "Pico médio com Mamba ativado (MB)",
    "Desvio padrão com Mamba ativado (MB)",
    "Pico médio com Mamba desativado (MB)",
    "Desvio padrão com Mamba desativado (MB)",
    "Aumento percentual médio por amostra (%)",
    "Desvio padrão do aumento percentual por amostra (%)",
)
SAMPLE_TABLE_COLUMNS = (
    "Conjunto de dados",
    "Modelo",
    "Amostra",
    "Ruído (razão)",
    "Geometria",
    "Épocas alvo",
    "Pico com Mamba ativado (MB)",
    "Pico com Mamba desativado (MB)",
    "Aumento percentual por amostra (%)",
)
FAILURE_TABLE_COLUMNS = (
    "Conjunto de dados",
    "Modelo",
    "Amostra",
    "Mamba",
    "Erro",
)


def _memory_value(value: Any) -> float | None:
    try:
        memory_mb = float(value)
    except (TypeError, ValueError):
        return None
    return memory_mb if math.isfinite(memory_mb) and memory_mb > 0.0 else None


def _cleanup_cuda_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _measure_without_mamba(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    model_name: str,
    device: torch.device,
    warmup_runs: int,
) -> float:
    if device.type != "cuda":
        raise RuntimeError("A comparação de memória dos modelos Mamba exige CUDA.")

    checkpoint_path = Path(str(row["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint ausente: {checkpoint_path}")
    checkpoint = load_checkpoint_low_memory(checkpoint_path)

    noise_value = row.get("noise_std_ratio", 0.0)
    noise_ratio = 0.0 if pd.isna(noise_value) else float(noise_value)
    run_config = apply_evaluation_noise(
        build_run_config(dataset_name, model_name),
        EvaluationJob(
            name="mamba-memory-comparison",
            model_names=(model_name,),
            dataset_names=(dataset_name,),
            noise_std_ratio=noise_ratio,
        ),
    )
    run_config["seed"] = int(run_config.get("seed", 42))
    saved_physics = checkpoint.get("physics_config")
    if saved_physics is not None:
        run_config["physics"] = saved_physics

    torch.manual_seed(run_config["seed"])
    data = prepare_data(
        run_config,
        torch.device("cpu"),
        sample_index=int(row["sample_id"]),
    )
    hyperparameters = checkpoint_hyperparameters(checkpoint, model_name)
    hyperparameters.update({switch: False for switch in MAMBA_SWITCHES})

    model = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_model(model_name, hyperparameters, device)

        checkpoint_state = checkpoint["model_state_dict"]
        model_state = model.state_dict()
        shared_state = {}
        unexpected_keys = []
        mismatched_shapes = []
        for name, tensor in checkpoint_state.items():
            if name not in model_state:
                if ".mamba." not in name and not name.startswith("mamba."):
                    unexpected_keys.append(name)
            elif tuple(tensor.shape) != tuple(model_state[name].shape):
                mismatched_shapes.append(name)
            else:
                shared_state[name] = tensor

        if unexpected_keys or mismatched_shapes:
            raise ValueError(
                "O checkpoint não corresponde à variante sem Mamba: "
                f"parâmetros inesperados={unexpected_keys}, "
                f"formas incompatíveis={mismatched_shapes}."
            )
        missing_keys = set(model_state).difference(shared_state)
        if missing_keys:
            raise ValueError(
                "A variante sem Mamba não recebeu pesos compartilhados para: "
                + ", ".join(sorted(missing_keys))
            )
        model.load_state_dict(shared_state, strict=True)
        model.eval()

        sample = prepare_training_sample(data, device)
        cnn_input = sample["cnn_input"]
        with torch.inference_mode():
            for _ in range(warmup_runs):
                model(cnn_input)
            model(cnn_input)
            torch.cuda.synchronize(device)
        return peak_gpu_memory_mb(device)
    finally:
        _cleanup_cuda_model(model)


def benchmark_mamba_memory_variants(
    scenario_results: Mapping[str, Any],
    dataset_names: Sequence[str],
    *,
    device: torch.device | None = None,
    warmup_runs: int = 1,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara inferência com Mamba ativo e inativo em checkpoints e amostras pareados.

    Os picos com Mamba ativo vêm da avaliação de cenários já executada. Os picos
    sem Mamba são medidos após substituir os blocos Mamba por módulos identidade
    e carregar do checkpoint os pesos convolucionais compartilhados. O aumento
    percentual é calculado para cada par de amostras antes de agregar. Não há
    retreinamento.
    """
    if warmup_runs < 0:
        raise ValueError("warmup_runs deve ser maior ou igual a zero.")
    eval_device = device or resolve_device("cuda_if_available")
    observations: dict[tuple[str, str, str], list[float]] = {}
    sample_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        result = scenario_results.get(dataset_name)
        if result is None:
            continue
        output = result.output
        for manifest_row in output.manifest:
            model_name = str(manifest_row.get("model_name", ""))
            if (
                model_name in MAMBA_MEMORY_MODELS
                and manifest_row.get("status") != "ok"
            ):
                failures.append({
                    "Conjunto de dados": dataset_name,
                    "Modelo": model_label(model_name),
                    "Amostra": manifest_row.get("sample_id", ""),
                    "Mamba": "Ativado",
                    "Erro": manifest_row.get("message", manifest_row.get("status", "falha")),
                })

        for row in output.rows:
            model_name = str(row.get("model_name", ""))
            if model_name not in MAMBA_MEMORY_MODELS or row.get("status", "ok") != "ok":
                continue
            active_memory = _memory_value(row.get("peak_gpu_memory_mb"))
            if active_memory is None:
                failures.append({
                    "Conjunto de dados": dataset_name,
                    "Modelo": model_label(model_name),
                    "Amostra": row.get("sample_id", ""),
                    "Mamba": "Ativado",
                    "Erro": "A avaliação existente não registrou pico de memória GPU.",
                })
                continue

            if eval_device.type != "cuda":
                failures.append({
                    "Conjunto de dados": dataset_name,
                    "Modelo": model_label(model_name),
                    "Amostra": row.get("sample_id", ""),
                    "Mamba": "Desativado",
                    "Erro": "A comparação sem Mamba exige CUDA.",
                })
                continue

            try:
                inactive_memory = _measure_without_mamba(
                    row,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    device=eval_device,
                    warmup_runs=warmup_runs,
                )
            except Exception as error:
                failures.append({
                    "Conjunto de dados": dataset_name,
                    "Modelo": model_label(model_name),
                    "Amostra": row.get("sample_id", ""),
                    "Mamba": "Desativado",
                    "Erro": str(error),
                })
                continue

            for variant, memory_mb in (
                ("Ativado", active_memory),
                ("Desativado", inactive_memory),
            ):
                key = (dataset_name, model_name, variant)
                observations.setdefault(key, []).append(memory_mb)
            sample_rows.append({
                "Conjunto de dados": dataset_name,
                "Modelo": model_label(model_name),
                "Amostra": row.get("sample_id", ""),
                "Ruído (razão)": row.get("noise_std_ratio", ""),
                "Geometria": row.get("receiver_geometry", ""),
                "Épocas alvo": row.get("epochs_target", ""),
                "Pico com Mamba ativado (MB)": active_memory,
                "Pico com Mamba desativado (MB)": inactive_memory,
                "Aumento percentual por amostra (%)": (
                    (active_memory / inactive_memory - 1.0) * 100.0
                ),
            })

    table_rows = []
    for dataset_name in dataset_names:
        for model_name in MAMBA_MEMORY_MODELS:
            table_row = {
                "Conjunto de dados": dataset_name,
                "Modelo": model_label(model_name),
            }
            for variant, label in (
                ("Ativado", "ativado"),
                ("Desativado", "desativado"),
            ):
                values = observations.get((dataset_name, model_name, variant), [])
                mean_column = f"Pico médio com Mamba {label} (MB)"
                std_column = f"Desvio padrão com Mamba {label} (MB)"
                mean = float(np.mean(values)) if values else np.nan
                table_row[mean_column] = mean
                table_row[std_column] = (
                    float(np.std(values, ddof=0)) if values else np.nan
                )
            percentages = [
                sample["Aumento percentual por amostra (%)"]
                for sample in sample_rows
                if sample["Conjunto de dados"] == dataset_name
                and sample["Modelo"] == model_label(model_name)
            ]
            table_row["Comparações pareadas (n)"] = len(percentages)
            table_row["Aumento percentual médio por amostra (%)"] = (
                float(np.mean(percentages)) if percentages else np.nan
            )
            table_row["Desvio padrão do aumento percentual por amostra (%)"] = (
                float(np.std(percentages, ddof=0)) if percentages else np.nan
            )
            table_rows.append(table_row)

    memory_table = pd.DataFrame(table_rows, columns=MEMORY_TABLE_COLUMNS)
    sample_table = pd.DataFrame(sample_rows, columns=SAMPLE_TABLE_COLUMNS)
    failure_table = pd.DataFrame(failures, columns=FAILURE_TABLE_COLUMNS)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        memory_table.to_csv(output_path, index=False)
        sample_output_path = output_path.with_name(
            f"{output_path.stem}_samples{output_path.suffix}"
        )
        sample_table.to_csv(sample_output_path, index=False)
    return memory_table, failure_table
