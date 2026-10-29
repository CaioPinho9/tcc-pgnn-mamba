"""Funções reutilizáveis de avaliação de checkpoints específicos por dataset.

    Os notebooks de treinamento e de avaliação usam o mesmo fluxo de preparação
    dos dados e construção do modelo. Este módulo reúne as etapas próprias da
    avaliação (busca de checkpoints, inferência, métricas e tabelas de saída),
    permitindo avaliar cada dataset sem depender do notebook.
"""

from __future__ import annotations

import gc
import csv
import json
import os
import platform
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config.config import (
    DEFAULT_CHECKPOINT_ROOT,
    EVALUATION_ROOT,
    MODEL_NAMES,
    MODEL_CONFIGS,
    TRAINING_LOGS_ROOT,
    config_for_dataset,
    load_training_config,
    resolve_physics_config,
    resolve_device,
)
from data.dataset import configured_sample_indices, prepare_data, prepare_training_sample
from data.seismic_noise import noise_condition_label
from helpers.checkpointing import (
    CheckpointArchitectureError,
    checkpoint_hyperparameters,
    read_checkpoint,
)
from helpers.metrics import (
    compute_mae,
    compute_ssim,
)
from helpers.runtime import peak_gpu_memory_mb, reset_peak_memory, synchronize_if_cuda
from network.models import build_model, is_mamba_model
from transformation.min_max_normalizer import MinMaxNormalizer
from visualization.charts.metrics import plot_grouped_scenario_metric_charts
from visualization.charts.performance import plot_scenario_runtime_charts
from visualization.charts.samples import plot_sample_comparison
from visualization.charts.scenarios import plot_scenario_charts
from visualization.charts.training import plot_training_losses
from visualization.charts.velocity import (
    plot_velocity_profiles,
    plot_velocity_scenario_comparison,
    reflow_cached_velocity_scenario_figure,
)


_DEFAULT_INFERENCE_WARMUP_RUNS = 5
_DEFAULT_INFERENCE_TIMED_RUNS = 30

MODEL_LABELS = {
    "pgnn_mardan_2024": "PGNN (Mardan 2024)",
    "pgann_mardan_2024": "PGANN (Mardan 2024)",
    "unet": "U-Net",
    "unetmamba": "U-Net Mamba",
    "mamba": "Mamba",
}

METRIC_COLUMNS = (
    "mse",
    "mae",
    "ssim",
)
METRIC_LABELS = {
    "mse": "MSE",
    "mae": "MAE",
    "ssim": "SSIM",
}
EVALUATION_CHECKPOINT = "best"  # Set to "last" to evaluate the final checkpoint.
_SAMPLE_SUFFIX = re.compile(
    rf"_s(?P<sample_id>\d+)_{re.escape(EVALUATION_CHECKPOINT)}$"
)


class EvaluationError(RuntimeError):
    """Classe base para falhas esperadas durante a avaliação."""

    status = "failed"


class IncompatibleCheckpointError(EvaluationError):
    """Indica que um checkpoint não pode processar o dataset selecionado."""
    status = "incompatible"


class CorruptCheckpointError(EvaluationError):
    """Indica que não foi possível ler o arquivo do checkpoint."""

    status = "corrupt_checkpoint"


@dataclass(frozen=True)
class EvaluationJob:
    """Uma tarefa independente de avaliação de dataset e modelo."""

    name: str
    model_names: tuple[str, ...]
    dataset_names: tuple[str, ...] | None = None
    checkpoint_dirs: tuple[str | Path, ...] = ()
    use_only_checkpoint_dirs: bool = False
    output_dir: str | Path | None = None
    sample_indices: tuple[int, ...] | None = None
    expected_untrained_model_names: tuple[str, ...] = ()
    noise_std_ratio: float | None = None
    input_noise_std_ratio: float | None = None
    observed_noise_std_ratio: float | None = None


@dataclass(frozen=True)
class EvaluationRun:
    model_name: str
    dataset_name: str
    sample_index: int
    checkpoint_path: Path
    run_name: str
    model_label: str
    noise_label: str = "clean"


@dataclass
class EvaluationOutput:
    job: EvaluationJob
    rows: list[dict[str, Any]] = field(default_factory=list)
    manifest: list[dict[str, Any]] = field(default_factory=list)
    predictions_by_sample: dict[str, dict[int, dict[str, np.ndarray]]] = field(default_factory=dict)
    true_velocity_by_sample: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ScenarioEvaluationResult:
    """Resultados, tabelas e figuras produzidos para um dataset."""

    output: EvaluationOutput
    factor_table: pd.DataFrame
    presentation_table: pd.DataFrame
    factor_table_path: Path
    presentation_table_path: Path
    map_figures: list[Path]
    metric_figures: list[Path]
    performance_figures: list[Path]
    reused_existing: bool = False


def model_label(model_name: str) -> str:
    """Retorna o rótulo de exibição estável usado em CSVs e gráficos."""
    return MODEL_LABELS.get(model_name, model_name)


def load_model_config(model_name: str) -> dict[str, Any]:
    """Carrega a configuração de treinamento e checkpoints do próprio modelo."""
    try:
        config_path = MODEL_CONFIGS[model_name]
    except KeyError as exc:
        raise ValueError(f"Modelo desconhecido {model_name!r}") from exc
    return load_training_config(config_path)


def load_evaluation_config(job: EvaluationJob) -> dict[str, Any]:
    """Resolve os datasets da tarefa, usando a configuração do modelo apenas como alternativa.

        Os datasets informados explicitamente na tarefa são prioritários, para que
        as tarefas de uma execução não dependam do modelo que aparece primeiro na lista.
        As configurações efetivas de cada modelo são carregadas separadamente por
        :func:`build_run_config`.
    """
    if not job.model_names:
        raise ValueError(f"O trabalho de avaliação {job.name!r} não possui modelos")
    dataset_names = job.dataset_names
    if not dataset_names:
        config = load_model_config(job.model_names[0])
        dataset_names = config.get("dataset_names")
    if not dataset_names:
        raise ValueError(f"Nenhum dataset configurado para o trabalho de avaliação {job.name!r}")
    return {"dataset_names": [str(name) for name in dataset_names]}


def build_run_config(dataset_name: str, model_name: str) -> dict[str, Any]:
    """Carrega a configuração de um modelo e aplica a física e as opções específicas do dataset."""
    run_config = config_for_dataset(
        load_model_config(model_name),
        dataset_name,
        model_name=model_name,
    )
    run_config = deepcopy(run_config)
    run_config["dataset_name"] = dataset_name
    return run_config


def apply_evaluation_noise(
        run_config: dict[str, Any],
        job: EvaluationJob,
) -> dict[str, Any]:
    """Aplica a condição de ruído explícita da tarefa à configuração de um modelo."""
    if job.noise_std_ratio is not None and (
            job.input_noise_std_ratio is not None
            or job.observed_noise_std_ratio is not None
    ):
        raise ValueError(
            "EvaluationJob deve usar noise_std_ratio ou os campos de ruído "
            "de entrada/observação, não ambos."
        )

    updated = deepcopy(run_config)
    if job.noise_std_ratio is not None:
        updated.setdefault("training", {})["seismic_input_noise_std_ratio"] = job.noise_std_ratio
        updated.setdefault("training", {})["seismic_observed_noise_std_ratio"] = job.noise_std_ratio
    else:
        if job.input_noise_std_ratio is not None:
            updated.setdefault("training", {})["seismic_input_noise_std_ratio"] = job.input_noise_std_ratio
        if job.observed_noise_std_ratio is not None:
            updated.setdefault("training", {})["seismic_observed_noise_std_ratio"] = job.observed_noise_std_ratio
    return updated


def evaluation_noise_label(run_config: dict[str, Any]) -> str:
    training = run_config.get("training", {})
    return noise_condition_label(
        training.get("seismic_input_noise_std_ratio"),
        training.get("seismic_observed_noise_std_ratio"),
    )


def dataset_noise_label(dataset_name: str, noise_label: str) -> str:
    return f"{dataset_name}-{noise_label}"


def resolve_checkpoint_dirs(
        config: dict[str, Any],
        model_names: Sequence[str],
        extra_dirs: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    """Reúne as pastas de checkpoints configuradas, locais e informadas explicitamente."""
    candidates: list[str | Path] = list(extra_dirs)
    candidates.append(DEFAULT_CHECKPOINT_ROOT)

    checkpointing = config.get("checkpointing", {})
    if checkpointing.get("dir"):
        candidates.append(checkpointing["dir"])
    # Inclua as raízes de configuração de cada modelo selecionado para que a
    # busca por checkpoints não dependa das configurações de outro modelo.
    for model_name in model_names:
        model_config_path = MODEL_CONFIGS.get(model_name)
        if model_config_path is None or not model_config_path.exists():
            continue
        model_config = load_training_config(model_config_path)
        model_checkpointing = model_config.get("checkpointing", {})
        if model_checkpointing.get("dir"):
            candidates.append(model_checkpointing["dir"])
    resolved: list[Path] = []
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


def checkpoint_patterns(
        model_name: str,
        dataset_name: str,
        noise_label: str = "clean",
) -> tuple[str, ...]:
    """Retorna os padrões de nome de checkpoint aceitos para um modelo e dataset."""
    run_suffix = "" if noise_label == "clean" else f"_{noise_label}"
    if noise_label == "clean":
        return (
            f"{model_name}_{dataset_name}_s*_{EVALUATION_CHECKPOINT}.pt",
            f"{model_name}_{dataset_name}_clean*_s*_{EVALUATION_CHECKPOINT}.pt",
        )
    return (
        f"{model_name}_{dataset_name}{run_suffix}*_s*_{EVALUATION_CHECKPOINT}.pt",
    )


def find_checkpoint_paths(
        model_name: str,
        dataset_name: str,
        checkpoint_dirs: Sequence[Path],
        noise_label: str = "clean",
) -> list[Path]:
    """Busca os checkpoints uma vez, preservando a ordem determinística das pastas."""
    found: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        for pattern in checkpoint_patterns(model_name, dataset_name, noise_label):
            for path in sorted(checkpoint_dir.glob(pattern)):
                if path.is_file() and path not in found:
                    found.append(path)
    return found


def parse_sample_index(checkpoint_path: Path) -> int | None:
    """Lê o identificador da amostra no checkpoint de avaliação selecionado."""
    match = _SAMPLE_SUFFIX.search(checkpoint_path.stem)
    return int(match.group("sample_id")) if match else None


def load_checkpoint_low_memory(path: Path) -> dict[str, Any]:
    """Carrega um checkpoint com mapeamento de memória (mmap), se a versão instalada do PyTorch aceitar."""
    try:
        return read_checkpoint(path, mmap=True)
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise CorruptCheckpointError(
            f"Não foi possível ler o checkpoint {path}: {error}"
        ) from error


def count_checkpoint_parameters(state_dict: dict[str, Any]) -> int:
    """Conta os elementos dos tensores treináveis e ignora os buffers de normalização."""
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return sum(
        tensor.numel()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor) and not name.endswith(buffer_suffixes)
    )


def dataset_sample_ids(
        run_config: dict[str, Any],
        requested_sample_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve identificadores determinísticos de amostras conforme a política de treinamento."""
    if requested_sample_indices is not None:
        return tuple(int(value) for value in requested_sample_indices)
    return tuple(int(value) for value in configured_sample_indices(run_config))


def discover_runs(
        run_config: dict[str, Any],
        model_names: Sequence[str],
        checkpoint_dirs: Sequence[Path],
        sample_indices: Sequence[int] | None = None,
        expected_untrained_model_names: Sequence[str] = (),
) -> tuple[list[EvaluationRun], list[dict[str, Any]]]:
    """Monta execuções por amostra e registros explícitos para checkpoints ausentes."""
    dataset_name = str(run_config["dataset_name"])
    noise_label = evaluation_noise_label(run_config)
    selected_sample_ids = dataset_sample_ids(run_config, sample_indices)
    runs: list[EvaluationRun] = []
    missing: list[dict[str, Any]] = []

    for model_name in model_names:
        if model_name not in MODEL_LABELS:
            raise ValueError(f"Modelo de avaliação desconhecido {model_name!r}")
        paths = find_checkpoint_paths(model_name, dataset_name, checkpoint_dirs, noise_label)
        if not paths:
            status = (
                "not_trained"
                if model_name in expected_untrained_model_names
                else "missing_checkpoint"
            )
            missing.append({
                "model_name": model_name,
                "model_label": model_label(model_name),
                "dataset_name": dataset_name,
                "noise_label": noise_label,
                "dataset_noise": dataset_noise_label(dataset_name, noise_label),
                "sample_id": "",
                "status": status,
                "checkpoint_path": "",
                "message": "; ".join(
                    str(path / pattern)
                    for path in checkpoint_dirs
                    for pattern in checkpoint_patterns(model_name, dataset_name, noise_label)
                ),
            })
            continue

        for path in paths:
            parsed_sample_id = parse_sample_index(path)
            if parsed_sample_id is not None:
                run_sample_ids = (
                    (parsed_sample_id,)
                    if parsed_sample_id in selected_sample_ids
                    else ()
                )
            else:
                run_sample_ids = selected_sample_ids
            for sample_id in run_sample_ids:
                if sample_id is None:
                    continue
                runs.append(EvaluationRun(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    noise_label=noise_label,
                    sample_index=int(sample_id),
                    checkpoint_path=path,
                    run_name=path.stem.removesuffix(f"_{EVALUATION_CHECKPOINT}"),
                    model_label=model_label(model_name),
                ))
    return runs, missing


def build_evaluation_model(
        run: EvaluationRun,
        checkpoint: dict[str, Any],
        run_config: dict[str, Any],
        data: dict[str, Any],
        device: torch.device,
) -> torch.nn.Module:
    """Constrói um modelo usando a mesma fábrica do treinamento."""
    if run.dataset_name == "Marmousi":
        saved_physics = checkpoint.get("physics_config")
        if saved_physics is None or {
            key: value
            for key, value in saved_physics.items()
            if key != "checkpoint_time_steps"
        } != {
            key: value
            for key, value in data["physics_config"].items()
            if key != "checkpoint_time_steps"
        }:
            raise IncompatibleCheckpointError(
                "O checkpoint Marmousi possui física de fonte/aquisição ausente ou incompatível. "
                "Treine-o novamente com as observações Marmousi atuais."
            )
    try:
        hyperparameters = checkpoint_hyperparameters(checkpoint, run.model_name)
        return build_model(run.model_name, hyperparameters, device)
    except CheckpointArchitectureError as error:
        raise IncompatibleCheckpointError(str(error)) from error
    except KeyError as error:
        raise IncompatibleCheckpointError(
            f"O checkpoint {run.run_name} não contém o parâmetro de arquitetura {error}."
        ) from error


def _is_cuda_memory_error(error: RuntimeError) -> bool:
    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        (out_of_memory and isinstance(error, out_of_memory))
        or "out of memory" in str(error).lower()
        or "cublas_status_alloc_failed" in str(error).lower()
    )


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_normalizers(
        checkpoint: dict[str, Any],
        hyperparameters: dict[str, Any],
        eval_device: torch.device,
) -> tuple[MinMaxNormalizer, MinMaxNormalizer]:
    normalizer_state = checkpoint.get("normalizer_state")
    if normalizer_state is None:
        raise IncompatibleCheckpointError(
            "O checkpoint não contém normalizer_state; este avaliador aceita apenas "
            "o formato compartilhado de checkpoint guiado por física para este modelo."
        )
    input_normalizer = MinMaxNormalizer.from_state(normalizer_state, eval_device)
    velocity_state = checkpoint.get("velocity_normalizer_state") or {
        "data_min": float(hyperparameters.get("m_min", 1500.0)),
        "data_max": float(hyperparameters.get("m_max", 4500.0)),
        "target_min": 0.0,
        "target_max": 1.0,
    }
    velocity_normalizer = MinMaxNormalizer.from_state(velocity_state, eval_device)
    return input_normalizer, velocity_normalizer


def predict_checkpoint(
        run: EvaluationRun,
        checkpoint: dict[str, Any],
        run_config: dict[str, Any],
        data: dict[str, Any],
        device: torch.device,
        *,
        warmup_runs: int = _DEFAULT_INFERENCE_WARMUP_RUNS,
        timed_runs: int = _DEFAULT_INFERENCE_TIMED_RUNS,
        benchmark_environment: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], float | None]:
    """Mede latência forward após aquecimento e retorna a mediana das repetições."""
    if warmup_runs < 0:
        raise ValueError("warmup_runs deve ser maior ou igual a zero.")
    if timed_runs < 1:
        raise ValueError("timed_runs deve ser maior que zero.")

    hyperparameters = checkpoint.get("hyperparameters") or {}
    eval_devices = [device]
    # O mamba-ssm usa kernels exclusivos de CUDA; tentar esses modelos na CPU
    # transforma um erro útil de CUDA na falha enganosa
    # "Expected x.is_cuda()".
    if device.type == "cuda" and not is_mamba_model(run.model_name):
        eval_devices.append(torch.device("cpu"))
    if is_mamba_model(run.model_name) and device.type != "cuda":
        raise EvaluationError("A avaliação do Mamba exige um dispositivo CUDA.")

    _clear_memory()

    last_error: RuntimeError | None = None
    peak_memory_mb = 0.0
    for eval_device in eval_devices:
        model: torch.nn.Module | None = None
        reset_peak_memory(eval_device)
        try:
            model = build_evaluation_model(run, checkpoint, run_config, data, eval_device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            input_normalizer, velocity_normalizer = _checkpoint_normalizers(
                checkpoint,
                hyperparameters,
                eval_device,
            )
            training_sample = prepare_training_sample(data, eval_device)
            with torch.inference_mode():
                cnn_input = training_sample["cnn_input"]
                for _ in range(warmup_runs):
                    prediction = model(cnn_input)
                synchronize_if_cuda(eval_device)
                latency_samples_ms: list[float] = []
                for _ in range(timed_runs):
                    synchronize_if_cuda(eval_device)
                    started = time.perf_counter()
                    prediction = model(cnn_input)
                    synchronize_if_cuda(eval_device)
                    latency_samples_ms.append((time.perf_counter() - started) * 1000.0)
                timing = {
                    "inference_time_ms": float(np.median(latency_samples_ms)),
                    "inference_time_mean_ms": float(np.mean(latency_samples_ms)),
                    "inference_time_std_ms": float(np.std(latency_samples_ms, ddof=0)),
                    "inference_time_p90_ms": float(np.percentile(latency_samples_ms, 90)),
                    "inference_time_samples_ms": json.dumps(latency_samples_ms),
                    "inference_warmup_runs": warmup_runs,
                    "inference_timed_runs": timed_runs,
                    "inference_benchmark_protocol": _inference_benchmark_protocol({
                        "warmup_runs": warmup_runs,
                        "timed_runs": timed_runs,
                    }),
                    "inference_device": str(eval_device),
                    "requested_device": str(device),
                    "benchmark_environment": benchmark_environment
                    or _inference_benchmark_environment(
                        device,
                        warmup_runs=warmup_runs,
                        timed_runs=timed_runs,
                    ),
                }
                peak_memory_mb = max(peak_memory_mb, peak_gpu_memory_mb(eval_device))
                predicted_velocity = velocity_normalizer.denormalize(prediction)

            predicted = predicted_velocity.detach().cpu().numpy()[0, 0]
            target_shape = tuple(int(value) for value in data["raw_y"].shape[-2:])
            if tuple(predicted.shape) != target_shape:
                raise IncompatibleCheckpointError(
                    f"A forma da saída do modelo {tuple(predicted.shape)} não corresponde "
                    f"à forma alvo {target_shape} para {run.dataset_name}."
                )
            measured_memory_mb = peak_memory_mb if eval_device.type == "cuda" else None
            return predicted, timing, measured_memory_mb
        except IncompatibleCheckpointError:
            raise
        except RuntimeError as error:
            last_error = error
            peak_memory_mb = max(peak_memory_mb, peak_gpu_memory_mb(eval_device))
            if (
                eval_device.type == "cuda"
                and _is_cuda_memory_error(error)
                and eval_device is not eval_devices[-1]
            ):
                continue
            raise EvaluationError(
                f"A inferência falhou para {run.run_name} em {eval_device}: {error}"
            ) from error
        finally:
            if model is not None:
                model.to("cpu")
            del model
            _clear_memory()

    if last_error is not None:
        raise last_error
    raise EvaluationError(f"Não foi possível avaliar {run.run_name}")


def score_prediction(
        run: EvaluationRun,
        checkpoint: dict[str, Any],
        prediction: np.ndarray,
        target: np.ndarray,
        timing: Mapping[str, Any],
        peak_memory_mb: float | None,
        physics_config: dict[str, Any],
) -> dict[str, Any]:
    """Calcula as métricas comuns de velocidade para uma predição bem-sucedida."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    mse = float(np.mean((prediction - target) ** 2))
    return {
        "dataset_name": run.dataset_name,
        "noise_label": run.noise_label,
        "dataset_noise": dataset_noise_label(run.dataset_name, run.noise_label),
        "sample_id": run.sample_index,
        "model_name": run.model_name,
        "model_label": run.model_label,
        "run_name": run.run_name,
        "checkpoint_path": str(run.checkpoint_path),
        "status": "ok",
        "epoch": checkpoint.get("epoch", ""),
        "num_parameters": count_checkpoint_parameters(checkpoint["model_state_dict"]),
        **timing,
        "peak_gpu_memory_mb": peak_memory_mb,
        "mse": mse,
        "mae": compute_mae(prediction, target),
        "ssim": compute_ssim(prediction, target),
    }


def _normalize_inference_benchmark(
        benchmark: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    benchmark = benchmark or {}
    warmup_runs = int(benchmark.get("warmup_runs", _DEFAULT_INFERENCE_WARMUP_RUNS))
    timed_runs = int(benchmark.get("timed_runs", _DEFAULT_INFERENCE_TIMED_RUNS))
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("O benchmark exige warmup_runs >= 0 e timed_runs >= 1.")
    return {"warmup_runs": warmup_runs, "timed_runs": timed_runs}


def _inference_benchmark_protocol(benchmark: Mapping[str, int]) -> str:
    return (
        f"warmup={benchmark['warmup_runs']}"
        f":repeats={benchmark['timed_runs']}"
    )


def _inference_benchmark_environment(
        device: torch.device,
        *,
        warmup_runs: int,
        timed_runs: int,
) -> str:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine() or "unknown-cpu"
    environment = {
        "protocol": _inference_benchmark_protocol({
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
        }),
        "requested_device": str(device),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "platform": platform.platform(),
    }
    return json.dumps(environment, sort_keys=True, separators=(",", ":"))


def _runtime_device_matches(row: Mapping[str, Any]) -> bool:
    requested = row.get("requested_device")
    actual = row.get("inference_device")
    if not requested or not actual:
        return True
    return str(requested).split(":", 1)[0] == str(actual).split(":", 1)[0]


def evaluate_dataset(
        run_config: dict[str, Any],
        model_names: Sequence[str],
        checkpoint_dirs: Sequence[Path],
        device: torch.device,
        *,
        sample_indices: Sequence[int] | None = None,
        expected_untrained_model_names: Sequence[str] = (),
        run_configs_by_model: dict[str, dict[str, Any]] | None = None,
) -> EvaluationOutput:
    """Avalia um dataset mantendo apenas uma amostra na memória por vez."""
    noise_label = evaluation_noise_label(run_config)
    training = run_config.get("training", {})
    job = EvaluationJob(
        name=dataset_noise_label(str(run_config["dataset_name"]), noise_label),
        model_names=tuple(model_names),
        dataset_names=(str(run_config["dataset_name"]),),
        checkpoint_dirs=tuple(checkpoint_dirs),
        input_noise_std_ratio=training.get("seismic_input_noise_std_ratio"),
        observed_noise_std_ratio=training.get("seismic_observed_noise_std_ratio"),
    )
    output = EvaluationOutput(job=job)
    runs, missing = discover_runs(
        run_config,
        model_names,
        checkpoint_dirs,
        sample_indices,
        expected_untrained_model_names,
    )
    output.manifest.extend(missing)

    for sample_id in sorted({run.sample_index for run in runs}):
        sample_runs = [run for run in runs if run.sample_index == sample_id]
        torch.manual_seed(int(run_config.get("seed", 42)))
        data = prepare_data(run_config, torch.device("cpu"), sample_index=sample_id)
        target = np.asarray(data["raw_y"][0, 0]).copy()
        output.true_velocity_by_sample.setdefault(run_config["dataset_name"], {})[sample_id] = target
        output.predictions_by_sample.setdefault(run_config["dataset_name"], {}).setdefault(sample_id, {})
        physics_config = data.get("physics_config") or resolve_physics_config(run_config)

        for run in sample_runs:
            checkpoint = None
            run_data = data
            run_config_for_model = (run_configs_by_model or {}).get(run.model_name, run_config)
            try:
                checkpoint = load_checkpoint_low_memory(run.checkpoint_path)
                saved_physics = checkpoint.get("physics_config")
                if (
                        run.dataset_name == "Marmousi"
                        and saved_physics is not None
                        and {
                            key: value
                            for key, value in saved_physics.items()
                            if key != "checkpoint_time_steps"
                        } != {
                            key: value
                            for key, value in data["physics_config"].items()
                            if key != "checkpoint_time_steps"
                        }
                ):
                    run_data_config = deepcopy(run_config_for_model)
                    run_data_config["physics"] = deepcopy(saved_physics)
                    torch.manual_seed(int(run_config_for_model.get("seed", 42)))
                    run_data = prepare_data(
                        run_data_config,
                        torch.device("cpu"),
                        sample_index=sample_id,
                    )
                run_physics_config = run_data.get("physics_config") or physics_config
                prediction, timing, peak_memory_mb = predict_checkpoint(
                    run,
                    checkpoint,
                    run_config_for_model,
                    run_data,
                    device,
                )
                row = score_prediction(
                    run,
                    checkpoint,
                    prediction,
                    target,
                    timing,
                    peak_memory_mb,
                    run_physics_config,
                )
                output.rows.append(row)
                output.predictions_by_sample[run_config["dataset_name"]][sample_id][run.model_label] = prediction
                output.manifest.append({**row, "message": ""})
            except EvaluationError as error:
                output.manifest.append({
                    "dataset_name": run.dataset_name,
                    "noise_label": run.noise_label,
                    "dataset_noise": dataset_noise_label(run.dataset_name, run.noise_label),
                    "sample_id": run.sample_index,
                    "model_name": run.model_name,
                    "model_label": run.model_label,
                    "run_name": run.run_name,
                    "status": error.status,
                    "checkpoint_path": str(run.checkpoint_path),
                    "message": str(error),
                })
            finally:
                del checkpoint
                if run_data is not data:
                    del run_data
                _clear_memory()
        del data
        _clear_memory()

    output.summary = summarize_metrics(output.rows)
    return output


def summarize_metrics(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Agrega as linhas bem-sucedidas por dataset, ruído e modelo."""
    frame = pd.DataFrame(list(rows))
    columns = [
        "dataset_name", "noise_label", "dataset_noise",
        "model_name", "model_label", "metric", "mean", "std", "median", "min", "max",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "status" in frame:
        frame = frame[frame["status"] == "ok"].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    summary_rows: list[dict[str, Any]] = []
    for (dataset_name, noise_label, condition, model_name, label), group in frame.groupby(
            ["dataset_name", "noise_label", "dataset_noise", "model_name", "model_label"],
            dropna=False,
    ):
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary_rows.append({
                "dataset_name": dataset_name,
                "noise_label": noise_label,
                "dataset_noise": condition,
                "model_name": model_name,
                "model_label": label,
                "metric": metric,
                "mean": values.mean() if not values.empty else np.nan,
                "std": values.std(ddof=0) if not values.empty else np.nan,
                "median": values.median() if not values.empty else np.nan,
                "min": values.min() if not values.empty else np.nan,
                "max": values.max() if not values.empty else np.nan,
            })
    return pd.DataFrame(summary_rows, columns=columns)


def evaluate_jobs(
        jobs: Sequence[EvaluationJob],
        device: torch.device | None = None,
) -> dict[str, EvaluationOutput]:
    """Executa tarefas de avaliação independentes e retorna uma saída para cada uma."""
    outputs: dict[str, EvaluationOutput] = {}
    eval_device = device or resolve_device("cuda_if_available")
    for job in jobs:
        config = load_evaluation_config(job)
        job_output = EvaluationOutput(job=job)
        if job.use_only_checkpoint_dirs:
            checkpoint_dirs = tuple(
                path if path.is_absolute() else Path.cwd() / path
                for path in (Path(value).expanduser() for value in job.checkpoint_dirs)
            )
        else:
            checkpoint_dirs = resolve_checkpoint_dirs(config, job.model_names, job.checkpoint_dirs)
        for dataset_name in config["dataset_names"]:
            run_configs_by_model = {
                model_name: apply_evaluation_noise(
                    build_run_config(dataset_name, model_name),
                    job,
                )
                for model_name in job.model_names
            }
            dataset_output = evaluate_dataset(
                run_configs_by_model[job.model_names[0]],
                job.model_names,
                checkpoint_dirs,
                eval_device,
                sample_indices=job.sample_indices,
                expected_untrained_model_names=job.expected_untrained_model_names,
                run_configs_by_model=run_configs_by_model,
            )
            job_output.rows.extend(dataset_output.rows)
            job_output.manifest.extend(dataset_output.manifest)
            for name, samples in dataset_output.predictions_by_sample.items():
                job_output.predictions_by_sample.setdefault(name, {}).update(samples)
            for name, samples in dataset_output.true_velocity_by_sample.items():
                job_output.true_velocity_by_sample.setdefault(name, {}).update(samples)
        job_output.summary = summarize_metrics(job_output.rows)
        outputs[job.name] = job_output
    return outputs


def save_evaluation_outputs(output: EvaluationOutput) -> Path:
    """Grava os CSVs de avaliação, incluindo uma tabela de resumo para cada métrica."""
    output_dir = Path(output.job.output_dir or EVALUATION_ROOT / output.job.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output.rows).to_csv(output_dir / "per_sample_metrics.csv", index=False)
    pd.DataFrame(output.manifest).to_csv(output_dir / "evaluation_manifest.csv", index=False)
    output.summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    summary_by_model = (
        output.summary.groupby(
            ["noise_label", "dataset_noise", "model_name", "model_label", "metric"],
            as_index=False,
            dropna=False,
        )["mean"]
        .mean()
        if not output.summary.empty
        else pd.DataFrame(columns=["noise_label", "dataset_noise", "model_name", "model_label", "metric", "mean"])
    )
    summary_by_model.to_csv(output_dir / "summary_metrics_by_model.csv", index=False)

    if output.summary.empty:
        summary_table = pd.DataFrame()
    else:
        summary_table = (
            output.summary.pivot_table(
                index=["dataset_name", "noise_label", "dataset_noise", "model_name", "model_label"],
                columns="metric",
                values="mean",
                aggfunc="first",
            )
            .reset_index()
        )
        summary_table.columns.name = None
    summary_table.to_csv(output_dir / "summary_table.csv", index=False)
    save_metric_tables(output, output_dir=output_dir)
    return output_dir


def _metric_specs(summary: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Retorna as métricas configuradas e acrescenta métricas futuras."""
    available = {str(value) for value in summary.get("metric", pd.Series(dtype=str))}
    ordered = [metric for metric in METRIC_COLUMNS if metric in available]
    ordered.extend(
        metric for metric in sorted(available)
        if metric not in ordered
    )
    return tuple((metric, METRIC_LABELS.get(metric, metric)) for metric in ordered)


def _ordered_model_names(model_names: Iterable[str]) -> list[str]:
    """Ordena modelos pela configuração global e mantém modelos extras ao final."""
    requested = list(dict.fromkeys(str(name) for name in model_names))
    present = set(requested)
    ordered = [name for name in MODEL_NAMES if name in present]
    ordered.extend(name for name in requested if name not in ordered)
    return ordered


def output_noise_labels(output: EvaluationOutput) -> tuple[str, ...]:
    labels = {
        str(row["noise_label"])
        for row in [*output.rows, *output.manifest]
        if row.get("noise_label")
    }
    if labels:
        return tuple(sorted(labels))
    if output.job.noise_std_ratio is not None:
        return (noise_condition_label(output.job.noise_std_ratio, output.job.noise_std_ratio),)
    return (noise_condition_label(output.job.input_noise_std_ratio, output.job.observed_noise_std_ratio),)


def save_metric_tables(
        output: EvaluationOutput,
        *,
        output_dir: str | Path | None = None,
) -> list[Path]:
    """Salva uma tabela CSV com estatísticas de resumo para cada métrica."""
    base_dir = Path(output_dir or output.job.output_dir or EVALUATION_ROOT / output.job.name)
    metric_dir = base_dir / "metrics" / "by_metric"
    summary = output.summary
    if summary.empty or "metric" not in summary.columns:
        return []

    saved: list[Path] = []
    columns = [
        "dataset_name",
        "noise_label",
        "dataset_noise",
        "model_name",
        "model_label",
        "mean",
        "std",
        "median",
        "min",
        "max",
    ]
    for metric, _ in _metric_specs(summary):
        table = summary[summary["metric"] == metric].copy()
        if table.empty:
            continue
        table = table[[column for column in columns if column in table.columns]]
        table = table.sort_values(["dataset_name", "noise_label", "model_name"])
        path = metric_dir / f"{metric}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        saved.append(path)
    return saved


def save_presentation_tables(
        output: EvaluationOutput,
) -> list[Path]:
    """Salva resumos de métricas e custos medidos durante a avaliação."""
    output_dir = Path(output.job.output_dir or EVALUATION_ROOT / output.job.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(output.rows)
    if rows.empty:
        metric_table = pd.DataFrame(columns=["noise_label", "dataset_noise", "model_name", "model_label", "metric", "mean", "std", "count", "dataset_count"])
        runtime_table = pd.DataFrame(columns=["dataset_name", "noise_label", "dataset_noise", "model_name", "model_label", "evaluated_samples", "mean_inference_time_ms", "std_inference_time_ms", "mean_peak_gpu_memory_mb", "std_peak_gpu_memory_mb"])
    else:
        if "status" in rows.columns:
            rows = rows[rows["status"] == "ok"].copy()
        metric_rows: list[dict[str, Any]] = []
        for (noise_label, dataset_noise, model_name, model_label, metric), group in rows.melt(
                id_vars=["model_name", "model_label", "dataset_name", "noise_label", "dataset_noise"],
                value_vars=[metric for metric in METRIC_COLUMNS if metric in rows.columns],
                var_name="metric",
                value_name="value",
        ).groupby(["noise_label", "dataset_noise", "model_name", "model_label", "metric"], dropna=False):
            values = pd.to_numeric(group["value"], errors="coerce").dropna()
            metric_rows.append({
                "noise_label": noise_label,
                "dataset_noise": dataset_noise,
                "model_name": model_name,
                "model_label": model_label,
                "metric": metric,
                "mean": values.mean() if not values.empty else np.nan,
                "std": values.std(ddof=0) if not values.empty else np.nan,
                "count": int(values.size),
                "dataset_count": int(group["dataset_name"].nunique()),
            })
        metric_table = pd.DataFrame(metric_rows)
        runtime_aggregations = {
            "evaluated_samples": ("sample_id", "count"),
            "dataset_count": ("dataset_name", "nunique"),
            "mean_inference_time_ms": ("inference_time_ms", "mean"),
            "std_inference_time_ms": ("inference_time_ms", lambda values: values.std(ddof=0)),
        }
        if "peak_gpu_memory_mb" in rows.columns:
            runtime_aggregations.update({
                "mean_peak_gpu_memory_mb": ("peak_gpu_memory_mb", "mean"),
                "std_peak_gpu_memory_mb": ("peak_gpu_memory_mb", lambda values: values.std(ddof=0)),
            })
        runtime_table = rows.groupby(
            ["dataset_name", "noise_label", "dataset_noise", "model_name", "model_label"],
            as_index=False,
            dropna=False,
        ).agg(**runtime_aggregations)

    paths = []
    metric_path = output_dir / "presentation_metrics_by_model.csv"
    metric_table.to_csv(metric_path, index=False)
    paths.append(metric_path)
    runtime_path = output_dir / "presentation_runtime_by_model.csv"
    runtime_table.to_csv(runtime_path, index=False)
    paths.append(runtime_path)
    return paths


def load_training_loss_history(
        output: EvaluationOutput,
        *,
        training_log_dir: str | Path = TRAINING_LOGS_ROOT / "v1" / "clean",
) -> dict[str, pd.DataFrame]:
    """Carrega e agrega as perdas de dados e do modelo para os modelos avaliados.

        Cada log de amostra contribui para a média e o desvio padrão de cada época.
        Os DataFrames retornados são organizados pelo nome do modelo e mantêm uma curva
        para cada dataset.
    """
    dataset_names = {
        str(name)
        for name in (output.job.dataset_names or ())
        if name
    }
    for row in [*output.rows, *output.manifest]:
        if row.get("dataset_name"):
            dataset_names.add(str(row["dataset_name"]))
    if not dataset_names:
        return {}

    log_dir = Path(training_log_dir)
    run_metadata = {
        str(row["run_name"]): row
        for row in [*output.rows, *output.manifest]
        if row.get("run_name") and row.get("model_name") in output.job.model_names
    }
    histories: dict[str, pd.DataFrame] = {}
    for model_name in output.job.model_names:
        records: list[pd.DataFrame] = []
        for run_name, metadata in run_metadata.items():
            if metadata.get("model_name") != model_name:
                continue
            for log_path in sorted(log_dir.rglob(f"{run_name}_training_log.csv")):
                try:
                    frame = pd.read_csv(log_path)
                except (OSError, pd.errors.ParserError):
                    continue
                required = {"epoch", "data_loss", "model_loss"}
                if not required.issubset(frame.columns):
                    continue
                frame = frame[["epoch", "data_loss", "model_loss"]].copy()
                frame.insert(0, "dataset_name", str(metadata.get("model_label", model_name)))
                noise_label = str(metadata.get("noise_label", "clean"))
                dataset_name = str(metadata.get("dataset_name", ""))
                frame.insert(1, "noise_label", noise_label)
                frame.insert(2, "dataset_noise", dataset_noise_label(dataset_name, noise_label))
                records.append(frame)

        if not records:
            continue
        history = pd.concat(records, ignore_index=True)
        for column in ("epoch", "data_loss", "model_loss"):
            history[column] = pd.to_numeric(history[column], errors="coerce")
        history = history.dropna(subset=["epoch", "data_loss", "model_loss"])
        if history.empty:
            continue
        histories[model_name] = (
            history.groupby(["dataset_name", "noise_label", "dataset_noise", "epoch"], as_index=False, sort=True)
            .agg(
                data_loss=("data_loss", "mean"),
                data_loss_std=("data_loss", lambda values: values.std(ddof=0)),
                model_loss=("model_loss", "mean"),
                model_loss_std=("model_loss", lambda values: values.std(ddof=0)),
            )
        )
    return histories


def save_loss_figures(
        output: EvaluationOutput,
        *,
        training_log_dir: str | Path = TRAINING_LOGS_ROOT / "v1" / "clean",
) -> list[Path]:
    """Salva um gráfico de perda dos dados e do modelo para cada modelo com log de treinamento."""
    histories = load_training_loss_history(output, training_log_dir=training_log_dir)
    figures_dir = Path(output.job.output_dir or EVALUATION_ROOT / output.job.name) / "figures" / "losses" / "by_model"
    saved: list[Path] = []
    for model_name in output.job.model_names:
        history = histories.get(model_name)
        if history is None or history.empty:
            continue
        labels = sorted(set(history["noise_label"].astype(str)))
        noise_suffix = labels[0] if len(labels) == 1 else "mixed"
        path = figures_dir / f"{model_name}_losses_{noise_suffix}.png"
        plot_training_losses(
            history,
            model_label=model_label(model_name),
            title=f"Perdas de treinamento - {model_label(model_name)} - {noise_suffix}",
            save_path=path,
        )
        if path.is_file():
            saved.append(path)
    return saved


def save_prediction_figures(
        output: EvaluationOutput,
        *,
        profile_positions: Sequence[float] = (0.25, 0.5, 0.75),
) -> list[Path]:
    """Salva mapas comparativos e perfis de velocidade por profundidade para predições bem-sucedidas.

        As posições dos perfis são frações da extensão horizontal do modelo. Para cada
        amostra, a figura do perfil é salva em ``figures/profiles`` junto à figura do mapa.
    """
    output_dir = Path(output.job.output_dir or EVALUATION_ROOT / output.job.name) / "figures"
    saved: list[Path] = []
    for dataset_name, samples in output.predictions_by_sample.items():
        run_config = build_run_config(
            dataset_name,
            output.job.model_names[0],
        )
        run_config = apply_evaluation_noise(run_config, output.job)
        physics_config = resolve_physics_config(run_config)
        for sample_id, predictions in samples.items():
            if not predictions:
                continue
            noise_labels = output_noise_labels(output)
            condition = (
                dataset_noise_label(dataset_name, noise_labels[0])
                if len(noise_labels) == 1
                else f"{dataset_name}-mixed_scenarios"
            )
            path = output_dir / f"{condition}_s{sample_id:04d}_model_comparison.png"
            plot_sample_comparison(
                sample_id,
                output.true_velocity_by_sample[dataset_name][sample_id],
                predictions,
                vmin=float(physics_config["velocity_min"]),
                vmax=float(physics_config["velocity_max"]),
                save_path=path,
            )
            saved.append(path)
            profile_path = output_dir / "profiles" / f"{condition}_s{sample_id:04d}_velocity_profiles.png"
            model_order = [
                model_label(name)
                for name in _ordered_model_names(output.job.model_names)
            ]
            model_order.extend(name for name in predictions if name not in model_order)
            plot_velocity_profiles(
                output.true_velocity_by_sample[dataset_name][sample_id],
                predictions,
                grid_spacing_m=float(physics_config["grid_spacing_m"]),
                title=f"{condition} - Amostra {sample_id} - Perfis de velocidade",
                save_path=profile_path,
                profile_positions=profile_positions,
                model_order=model_order,
            )
            saved.append(profile_path)
    return saved


# Avaliação dos cenários de treinamento.
_SCENARIO_SUFFIX = re.compile(
    r"^(?P<condition>clean|noise_(?P<noise_ratio>(?:m)?\d+(?:p\d+)?))"
    r"_epochs(?P<epochs>\d+)"
    r"_(?P<geometry>surface(?:_wells)?)"
    r"_s(?P<sample_id>\d{4})$"
)
_GEOMETRY_LABELS = {
    "surface": "superfície",
    "surface_wells": "superfície + poços",
}
_TRAINING_METRICS = ("data_loss", "model_loss", "total_loss")
_EVALUATION_METRICS = (
    "mse",
    "mae",
    "ssim",
    "inference_time_ms",
    "peak_gpu_memory_mb",
)
_PRESENTATION_METRICS = (
    "mse",
    "mae",
    "ssim",
)
_SCENARIO_METRIC_DIRECTIONS = {
    "mse": "lower",
    "mae": "lower",
    "ssim": "higher",
    "inference_time_ms": "lower",
    "peak_gpu_memory_mb": "lower",
}
_CHECKPOINT_LOSSES = (
    "epoch",
    "data_loss",
    "model_loss",
    "total_loss",
)


def _decode_noise_ratio(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _scenario_from_path(
        path: Path,
        model_name: str | None,
        dataset_name: str,
) -> dict[str, Any] | None:
    """Extrai os fatores do nome canônico de um checkpoint ou log de treinamento."""
    stem = path.stem
    for suffix in ("_training_log", f"_{EVALUATION_CHECKPOINT}"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    candidates = (model_name,) if model_name is not None else tuple(
        sorted(MODEL_NAMES, key=len, reverse=True)
    )
    matched_model = next(
        (
            candidate
            for candidate in candidates
            if stem.startswith(f"{candidate}_{dataset_name}_")
        ),
        None,
    )
    if matched_model is None:
        return None
    prefix = f"{matched_model}_{dataset_name}_"
    match = _SCENARIO_SUFFIX.fullmatch(stem[len(prefix):])
    if match is None:
        return None

    values = match.groupdict()
    condition = values["condition"]
    noise_ratio = (
        0.0
        if condition == "clean"
        else _decode_noise_ratio(values["noise_ratio"])
    )
    epochs = int(values["epochs"])
    geometry = values["geometry"]
    noise_text = "sem ruído" if noise_ratio == 0.0 else f"ruído {noise_ratio:g}"
    scenario_label = (
        f"{model_label(matched_model)} | {noise_text} | {epochs} épocas | "
        f"{_GEOMETRY_LABELS[geometry]}"
    )
    return {
        "model_name": matched_model,
        "model_label": model_label(matched_model),
        "noise_std_ratio": noise_ratio,
        "noise_label": condition,
        "epochs": epochs,
        "receiver_geometry": geometry,
        "sample_id": int(values["sample_id"]),
        "scenario_id": f"{matched_model}__{condition}__epochs_{epochs}__geometry_{geometry}",
        "scenario_label": scenario_label,
    }


def _version_root(root: str | Path, version: str) -> Path:
    path = Path(root).expanduser()
    return path if path.name == version else path / version


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=float)
    return float(numeric.mean()), float(numeric.std(ddof=0))


def build_training_factor_table(
        *,
        version: str = "v6",
        model_names: Sequence[str] | None = None,
        model_name: str | None = None,
        dataset_name: str = "Marmousi",
        checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
        training_log_root: str | Path = TRAINING_LOGS_ROOT,
) -> pd.DataFrame:
    """Resume cenários com checkpoints na raiz da versão e agrega seus logs."""
    selected_models = tuple(model_names or ((model_name,) if model_name else MODEL_NAMES))
    checkpoint_dir = _version_root(checkpoint_root, version)
    log_dir = _version_root(training_log_root, version)
    scenarios: dict[str, dict[str, Any]] = {}

    def get_scenario(
            path: Path,
            *,
            create: bool,
    ) -> tuple[str, dict[str, Any]] | None:
        factors = _scenario_from_path(path, None, dataset_name)
        if factors is None or factors["model_name"] not in selected_models:
            return None
        scenario_id = factors["scenario_id"]
        row = scenarios.get(scenario_id)
        if row is None:
            if not create:
                return None
            row = {
                **factors,
                "checkpoint_samples": set(),
                "training_log_samples": set(),
                "training_log_count": 0,
                "max_logged_epoch": None,
                "training_metric_values": {
                    metric: [] for metric in _TRAINING_METRICS
                },
                "checkpoint_metric_values": {
                    metric: [] for metric in _CHECKPOINT_LOSSES
                },
            }
            scenarios[scenario_id] = row
        return scenario_id, row

    if checkpoint_dir.is_dir():
        for path in sorted(
            checkpoint_dir.glob(f"*_{dataset_name}_*_{EVALUATION_CHECKPOINT}.pt")
        ):
            parsed = get_scenario(path, create=True)
            if parsed is not None:
                _, row = parsed
                factors = _scenario_from_path(path, None, dataset_name)
                row["checkpoint_samples"].add(factors["sample_id"])
                try:
                    checkpoint = read_checkpoint(path, mmap=True)
                    for metric in _CHECKPOINT_LOSSES:
                        value = checkpoint.get(metric)
                        if value is not None and np.isfinite(float(value)):
                            row["checkpoint_metric_values"][metric].append(float(value))
                    del checkpoint
                except (EOFError, OSError, RuntimeError, TypeError, ValueError):
                    continue

    if log_dir.is_dir():
        log_pattern = f"*_{dataset_name}_*_training_log.csv"
        for path in sorted(log_dir.rglob(log_pattern)):
            parsed = get_scenario(path, create=False)
            if parsed is None:
                continue
            _, row = parsed
            factors = _scenario_from_path(path, None, dataset_name)
            row["training_log_samples"].add(factors["sample_id"])
            row["training_log_count"] += 1
            try:
                with path.open(newline="", encoding="utf-8") as stream:
                    log_rows = list(csv.DictReader(stream))
                valid_rows = [item for item in log_rows if item.get("epoch", "").strip()]
                final_row = max(valid_rows, key=lambda item: int(item["epoch"]))
                epochs = [int(item["epoch"]) for item in valid_rows]
            except (OSError, ValueError, KeyError):
                final_row = {}
                epochs = []
            if epochs:
                last_epoch = max(epochs)
                current = row["max_logged_epoch"]
                row["max_logged_epoch"] = max(last_epoch, current or 0)
            for metric in _TRAINING_METRICS:
                value = final_row.get(metric, "").strip()
                if value:
                    try:
                        row["training_metric_values"][metric].append(float(value))
                    except ValueError:
                        continue

    columns = [
        "scenario_id",
        "model_name",
        "model_label",
        "noise_label",
        "noise_std_ratio",
        "epochs",
        "receiver_geometry",
        "geometry_description",
        "checkpoint_count",
        "checkpoint_samples",
        "training_log_count",
        "training_log_samples",
        "max_logged_epoch",
        "status",
        "checkpoint_selection",
        "evaluated_samples",
        "failed_evaluations",
        "evaluation_status",
    ]
    for metric in _TRAINING_METRICS:
        columns.extend((
            f"training_last_logged_{metric}_mean",
            f"training_last_logged_{metric}_std",
        ))
    for metric in _CHECKPOINT_LOSSES:
        columns.extend((f"checkpoint_{metric}_mean", f"checkpoint_{metric}_std"))
    for metric in _EVALUATION_METRICS:
        columns.extend((f"evaluation_{metric}_mean", f"evaluation_{metric}_std"))
    rows = []
    for row in scenarios.values():
        checkpoint_samples = sorted(row["checkpoint_samples"])
        log_samples = sorted(row["training_log_samples"])
        if checkpoint_samples and log_samples:
            status = "checkpoints e logs disponíveis"
        elif checkpoint_samples:
            status = "checkpoints disponíveis; log ausente"
        else:
            status = "log disponível; checkpoint ausente"
        output_row = {
            "scenario_id": row["scenario_id"],
            "model_name": row["model_name"],
            "model_label": row["model_label"],
            "noise_label": row["noise_label"],
            "noise_std_ratio": row["noise_std_ratio"],
            "epochs": row["epochs"],
            "receiver_geometry": row["receiver_geometry"],
            "geometry_description": _GEOMETRY_LABELS[row["receiver_geometry"]],
            "checkpoint_count": len(checkpoint_samples),
            "checkpoint_samples": ", ".join(map(str, checkpoint_samples)),
            "training_log_count": row["training_log_count"],
            "training_log_samples": ", ".join(map(str, log_samples)),
            "max_logged_epoch": row["max_logged_epoch"],
            "status": status,
            "checkpoint_selection": EVALUATION_CHECKPOINT,
            "evaluated_samples": 0,
            "failed_evaluations": 0,
            "evaluation_status": "pendente",
        }
        for metric in _TRAINING_METRICS:
            values = row["training_metric_values"][metric]
            mean, std = _mean_and_std(values) if values else (np.nan, np.nan)
            output_row[f"training_last_logged_{metric}_mean"] = mean
            output_row[f"training_last_logged_{metric}_std"] = std
        for metric in _CHECKPOINT_LOSSES:
            values = row["checkpoint_metric_values"][metric]
            mean, std = _mean_and_std(values) if values else (np.nan, np.nan)
            output_row[f"checkpoint_{metric}_mean"] = mean
            output_row[f"checkpoint_{metric}_std"] = std
        for metric in _EVALUATION_METRICS:
            output_row[f"evaluation_{metric}_mean"] = np.nan
            output_row[f"evaluation_{metric}_std"] = np.nan
        rows.append(output_row)
    rows.sort(key=lambda row: (
        row["noise_std_ratio"],
        row["epochs"],
        row["receiver_geometry"],
        row["model_name"],
    ))
    return pd.DataFrame(rows, columns=columns)


def add_scenario_evaluation_metrics(
        factor_table: pd.DataFrame,
        output: EvaluationOutput,
) -> pd.DataFrame:
    """Acrescenta à tabela as métricas de avaliação agregadas por cenário."""
    table = factor_table.copy()
    for column in ("evaluated_samples", "failed_evaluations", "evaluation_status"):
        if column not in table.columns:
            table[column] = 0 if column != "evaluation_status" else "pendente"
    for metric in _EVALUATION_METRICS:
        for statistic in ("mean", "std"):
            column = f"evaluation_{metric}_{statistic}"
            if column not in table.columns:
                table[column] = np.nan

    def scenario_key(row: dict[str, Any]) -> tuple[str, float, int, str] | None:
        epochs = row.get("epochs_target", row.get("epochs"))
        try:
            return (
                str(row["model_name"]),
                float(row["noise_std_ratio"]),
                int(epochs),
                str(row["receiver_geometry"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    successful: dict[tuple[str, float, int, str], list[dict[str, Any]]] = defaultdict(list)
    failed: dict[tuple[str, float, int, str], int] = defaultdict(int)
    for row in output.rows:
        key = scenario_key(row)
        if key is not None and row.get("status") == "ok":
            successful[key].append(row)
    for row in output.manifest:
        key = scenario_key(row)
        if key is not None and row.get("status") != "ok":
            failed[key] += 1

    for index, row in table.iterrows():
        key = (
            str(row["model_name"]),
            float(row["noise_std_ratio"]),
            int(row["epochs"]),
            str(row["receiver_geometry"]),
        )
        results = successful.get(key, [])
        sample_count = len({item.get("sample_id") for item in results})
        failed_count = failed.get(key, 0)
        table.at[index, "evaluated_samples"] = sample_count
        table.at[index, "failed_evaluations"] = failed_count
        expected_count = int(row.get("checkpoint_count", 0) or 0)
        if sample_count:
            table.at[index, "evaluation_status"] = (
                "calculada" if sample_count >= expected_count else "parcial"
            )
        elif failed_count:
            table.at[index, "evaluation_status"] = "falhou"
        for metric in _EVALUATION_METRICS:
            metric_results = results
            if metric == "inference_time_ms":
                metric_results = [item for item in results if _runtime_device_matches(item)]
            values = [
                float(item[metric])
                for item in metric_results
                if item.get(metric) is not None and np.isfinite(float(item[metric]))
            ]
            if values:
                mean, std = _mean_and_std(values)
                table.at[index, f"evaluation_{metric}_mean"] = mean
                table.at[index, f"evaluation_{metric}_std"] = std
    return table


def scenario_presentation_table(factor_table: pd.DataFrame) -> pd.DataFrame:
    """Retorna fatores, estado e métricas medidos para tabelas e gráficos.

    Perdas de treinamento e valores lidos dos checkpoints permanecem disponíveis
    na tabela completa, mas não fazem parte do resumo usado na apresentação.
    """
    columns = [
        "scenario_id",
        "model_name",
        "model_label",
        "noise_label",
        "noise_std_ratio",
        "epochs",
        "receiver_geometry",
        "geometry_description",
        "checkpoint_count",
        "checkpoint_samples",
        "evaluated_samples",
        "failed_evaluations",
        "evaluation_status",
    ]
    scenario_metrics = (*_PRESENTATION_METRICS, "inference_time_ms", "peak_gpu_memory_mb")
    for metric in scenario_metrics:
        columns.extend((
            f"evaluation_{metric}_mean",
            f"evaluation_{metric}_std",
            f"evaluation_{metric}_best",
        ))

    presentation = factor_table.copy()
    for column in columns:
        if column not in presentation:
            presentation[column] = "" if column.endswith("_best") else np.nan
    presentation = presentation[columns]
    checkpoint_counts = pd.to_numeric(presentation["checkpoint_count"], errors="coerce").fillna(0)
    presentation = presentation.loc[checkpoint_counts.gt(0)].copy()
    scenario_columns = ["noise_std_ratio", "epochs", "receiver_geometry"]
    for metric, direction in _SCENARIO_METRIC_DIRECTIONS.items():
        mean_column = f"evaluation_{metric}_mean"
        best_column = f"evaluation_{metric}_best"
        presentation[best_column] = ""
        values = pd.to_numeric(presentation[mean_column], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        if not valid.any():
            continue
        candidates = presentation.loc[valid, scenario_columns].copy()
        candidates["_metric_value"] = values.loc[valid]
        grouped = candidates.groupby(scenario_columns, dropna=False)["_metric_value"]
        best_values = grouped.transform("min" if direction == "lower" else "max")
        is_best = candidates["_metric_value"].eq(best_values)
        presentation.loc[candidates.index[is_best], best_column] = "↑" if direction == "higher" else "↓"
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    return (
        presentation.assign(
            _model_order=presentation["model_name"].map(model_order).fillna(len(model_order))
        )
        .sort_values(
            ["noise_std_ratio", "epochs", "receiver_geometry", "_model_order", "model_name"],
            kind="stable",
        )
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def aggregate_grouped_scenario_metrics(
        per_sample_rows: pd.DataFrame,
        metric_definitions: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Calcula médias por fator e modelo a partir das avaliações válidas por amostra."""
    group_columns = {
        "noise": ("noise_std_ratio", "Ruído"),
        "geometry": ("receiver_geometry", "Geometria"),
        "model": ("model_name", "Modelo"),
    }
    metric_keys = [str(item["key"]) for item in metric_definitions]
    columns = [
        "category_key", "category", "model_name", "model_label",
        "evaluated_samples", "model_count", "dataset_count", "scenario_count",
    ]
    for key in metric_keys:
        columns.extend((
            f"evaluation_{key}_mean",
            f"evaluation_{key}_std",
            f"evaluation_{key}_count",
            f"evaluation_{key}_best",
        ))
    empty_tables = {name: pd.DataFrame(columns=columns) for name in group_columns}
    if per_sample_rows.empty:
        return empty_tables

    rows = per_sample_rows.copy()
    if "status" in rows:
        rows = rows.loc[rows["status"].astype(str).eq("ok")].copy()
    required = {"dataset_name", "model_name", "noise_std_ratio", "receiver_geometry"}
    if not required.issubset(rows.columns):
        return empty_tables
    rows = rows.dropna(subset=list(required)).copy()
    if rows["dataset_name"].astype(str).nunique() > 1:
        raise ValueError("Group metrics must be calculated one dataset at a time.")
    metric_keys = [key for key in metric_keys if key in rows.columns]
    if not metric_keys or rows.empty:
        return empty_tables

    for key in metric_keys:
        numeric = pd.to_numeric(rows[key], errors="coerce")
        rows[key] = numeric.where(np.isfinite(numeric))
    rows["_scenario_id"] = rows[
        [column for column in (
            "dataset_name", "model_name", "noise_std_ratio", "epochs_target", "receiver_geometry"
        ) if column in rows.columns]
    ].astype("string").fillna("").agg("|".join, axis=1)

    def noise_category(value: Any) -> tuple[str, str]:
        ratio = float(value)
        key = f"{ratio:.12g}"
        label = "Sem ruído" if ratio == 0 else f"Ruído {key}".replace(".", ",")
        return key, label

    table_by_group: dict[str, pd.DataFrame] = {}
    for group_name, (source_column, _) in group_columns.items():
        output_records: list[dict[str, Any]] = []
        category_rows = rows.copy()
        if group_name == "noise":
            labels = category_rows[source_column].map(noise_category)
            category_rows["_category_key"] = labels.map(lambda item: item[0])
            category_rows["_category"] = labels.map(lambda item: item[1])
        elif group_name == "geometry":
            category_rows["_category_key"] = category_rows[source_column].astype(str)
            category_rows["_category"] = category_rows["_category_key"].map(
                lambda value: _GEOMETRY_LABELS.get(value, value.replace("_", " "))
            )
        else:
            category_rows["_category_key"] = category_rows[source_column].astype(str)
            category_rows["_category"] = category_rows["_category_key"].map(
                lambda value: model_label(value)
            )

        aggregation_columns = (
            ["_category_key"] if group_name == "model"
            else ["_category_key", "model_name"]
        )
        for group_values, group in category_rows.groupby(
            aggregation_columns, sort=False, dropna=False
        ):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            category_key = str(group_values[0])
            model_name = str(group["model_name"].iloc[0])
            first = group.iloc[0]
            record: dict[str, Any] = {
                "category_key": category_key,
                "category": str(first["_category"]),
                "model_name": model_name,
                "model_label": model_label(model_name),
                "evaluated_samples": int(group[metric_keys].notna().any(axis=1).sum()),
                "model_count": int(group["model_name"].nunique()),
                "dataset_count": int(group["dataset_name"].nunique()),
                "scenario_count": int(group["_scenario_id"].nunique()),
            }
            for key in metric_keys:
                values = group[key].dropna()
                record[f"evaluation_{key}_mean"] = float(values.mean()) if not values.empty else np.nan
                record[f"evaluation_{key}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
                record[f"evaluation_{key}_count"] = int(values.size)
                record[f"evaluation_{key}_best"] = ""
            output_records.append(record)

        table = pd.DataFrame.from_records(output_records, columns=columns)
        for definition in metric_definitions:
            key = str(definition["key"])
            mean_column = f"evaluation_{key}_mean"
            best_column = f"evaluation_{key}_best"
            if mean_column not in table:
                continue
            values = pd.to_numeric(table[mean_column], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            if valid.any():
                direction = str(definition.get("direction", _SCENARIO_METRIC_DIRECTIONS.get(key, "lower")))
                if group_name == "model":
                    best_values = pd.Series(
                        values[valid].max() if direction == "higher" else values[valid].min(),
                        index=table.index[valid],
                    )
                else:
                    comparisons = table.loc[valid, ["category_key"]].copy()
                    comparisons["_mean"] = values.loc[valid]
                    best_values = comparisons.groupby("category_key", dropna=False)["_mean"].transform(
                        "max" if direction == "higher" else "min"
                    )
                    best_values.index = comparisons.index
                table.loc[valid & values.eq(best_values), best_column] = "↑" if direction == "higher" else "↓"

        if group_name == "noise":
            table["_order"] = pd.to_numeric(table["category_key"], errors="coerce")
            table["_model_order"] = table["model_name"].map(
                {name: index for index, name in enumerate(MODEL_NAMES)}
            ).fillna(len(MODEL_NAMES))
            table = table.sort_values(["_order", "_model_order", "model_name"], kind="stable").drop(columns=["_order", "_model_order"])
        elif group_name == "geometry":
            geometry_order = {name: index for index, name in enumerate(_GEOMETRY_LABELS)}
            table["_order"] = table["category_key"].map(geometry_order).fillna(len(geometry_order))
            table["_model_order"] = table["model_name"].map(
                {name: index for index, name in enumerate(MODEL_NAMES)}
            ).fillna(len(MODEL_NAMES))
            table = table.sort_values(["_order", "category_key", "_model_order", "model_name"], kind="stable").drop(columns=["_order", "_model_order"])
        else:
            order = {name: index for index, name in enumerate(MODEL_NAMES)}
            table["_order"] = table["category_key"].map(order).fillna(len(order))
            table = table.sort_values(["_order", "category_key"], kind="stable").drop(columns="_order")
        table_by_group[group_name] = table.reset_index(drop=True)
    return table_by_group


def write_scenario_grouped_presentation_tables(
        presentation_config: Mapping[str, Any] | str | Path | None = None,
        *,
        evaluation_root: str | Path = EVALUATION_ROOT,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Path]]]:
    """Grava tabelas agrupadas de qualidade separadamente para cada dataset."""
    if presentation_config is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "evaluation_presentation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif isinstance(presentation_config, Mapping):
        config = dict(presentation_config)
    else:
        config = json.loads(Path(presentation_config).read_text(encoding="utf-8"))

    evaluation_root = Path(evaluation_root)
    version_root = evaluation_root / str(config["version"])
    tables_by_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    paths_by_dataset: dict[str, dict[str, Path]] = {}
    filename_pattern = config.get(
        "grouped_metric_table_filename", "presentation_grouped_by_{group}.csv"
    )
    for dataset in config.get("datasets", ()):
        dataset_name = str(dataset["name"])
        relative_dir = dataset.get(
            "output_dir",
            config["scenario_output_dir"].format(dataset=dataset_name),
        )
        source_path = version_root / relative_dir / "per_sample_metrics.csv"
        if source_path.is_file():
            try:
                per_sample_rows = pd.read_csv(source_path)
            except pd.errors.EmptyDataError:
                per_sample_rows = pd.DataFrame()
        else:
            per_sample_rows = pd.DataFrame()
        if "dataset_name" in per_sample_rows:
            per_sample_rows = per_sample_rows.loc[
                per_sample_rows["dataset_name"].astype(str).eq(dataset_name)
            ].copy()
        tables = aggregate_grouped_scenario_metrics(
            per_sample_rows,
            config.get("metrics", ()),
        )
        paths: dict[str, Path] = {}
        for group_name, table in tables.items():
            path = version_root / relative_dir / filename_pattern.format(
                dataset=dataset_name,
                group=group_name,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
            paths[group_name] = path
        tables_by_dataset[dataset_name] = tables
        paths_by_dataset[dataset_name] = paths
    return tables_by_dataset, paths_by_dataset


def aggregate_scenario_runtime_metrics(
        per_sample_rows: pd.DataFrame,
        dataset_order: Sequence[str] = (),
) -> pd.DataFrame:
    """Agrupa medianas de latência e picos de memória por dataset, modelo e geometria.

    Mean, median, standard deviation, and p90 describe the per-evaluation medians;
    ``evaluation_inference_repeat_std_ms_mean`` describes variation within repeats.
    """
    columns = [
        "dataset_name",
        "model_name",
        "model_label",
        "receiver_geometry",
        "geometry_description",
        "evaluated_samples",
        "inference_devices",
        "evaluation_inference_time_ms_mean",
        "evaluation_inference_time_ms_median",
        "evaluation_inference_time_ms_std",
        "evaluation_inference_time_ms_p90",
        "evaluation_inference_time_ms_count",
        "evaluation_inference_repeat_std_ms_mean",
        "evaluation_inference_time_ms_best",
        "evaluation_peak_gpu_memory_mb_mean",
        "evaluation_peak_gpu_memory_mb_std",
        "evaluation_peak_gpu_memory_mb_count",
        "evaluation_peak_gpu_memory_mb_best",
    ]
    if per_sample_rows.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "dataset_name", "model_name", "receiver_geometry",
        "inference_time_ms", "peak_gpu_memory_mb",
    }
    if not required.issubset(per_sample_rows.columns):
        return pd.DataFrame(columns=columns)

    rows = per_sample_rows.copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).eq("ok")].copy()
    rows["dataset_name"] = rows["dataset_name"].astype(str)
    rows = rows.dropna(subset=["dataset_name", "model_name", "receiver_geometry"])
    for metric in ("inference_time_ms", "peak_gpu_memory_mb"):
        values = pd.to_numeric(rows[metric], errors="coerce")
        rows[metric] = values.where(np.isfinite(values))
    if {"requested_device", "inference_device"}.issubset(rows.columns):
        compatible_device = rows.apply(_runtime_device_matches, axis=1)
        rows.loc[~compatible_device, "inference_time_ms"] = np.nan

    grouping = ["dataset_name", "model_name", "receiver_geometry"]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(grouping, dropna=False, sort=False):
        record = dict(zip(grouping, key))
        record["model_label"] = model_label(str(record["model_name"]))
        geometry = str(record["receiver_geometry"])
        record["geometry_description"] = _GEOMETRY_LABELS.get(
            geometry, geometry.replace("_", " ")
        )
        record["inference_devices"] = ", ".join(sorted(
            group.get("inference_device", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        ))
        record["evaluated_samples"] = int(group["inference_time_ms"].count())
        for metric in ("inference_time_ms", "peak_gpu_memory_mb"):
            values = group[metric].dropna()
            record[f"evaluation_{metric}_mean"] = (
                float(values.mean()) if not values.empty else np.nan
            )
            if metric == "inference_time_ms":
                record[f"evaluation_{metric}_median"] = (
                    float(values.median()) if not values.empty else np.nan
                )
                record[f"evaluation_{metric}_p90"] = (
                    float(np.percentile(values, 90)) if not values.empty else np.nan
                )
                repeat_std = pd.to_numeric(
                    group.loc[
                        group["inference_time_ms"].notna(), "inference_time_std_ms"
                    ]
                    if "inference_time_std_ms" in group.columns
                    else pd.Series(dtype=float),
                    errors="coerce",
                ).dropna()
                record["evaluation_inference_repeat_std_ms_mean"] = (
                    float(repeat_std.mean()) if not repeat_std.empty else np.nan
                )
            record[f"evaluation_{metric}_std"] = (
                float(values.std(ddof=0)) if not values.empty else np.nan
            )
            record[f"evaluation_{metric}_count"] = int(values.size)
            record[f"evaluation_{metric}_best"] = ""
        records.append(record)

    result = pd.DataFrame.from_records(records, columns=columns)
    for metric in ("inference_time_ms", "peak_gpu_memory_mb"):
        mean_column = f"evaluation_{metric}_mean"
        best_column = f"evaluation_{metric}_best"
        values = pd.to_numeric(result[mean_column], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        if valid.any():
            comparison = result.loc[
                valid, ["dataset_name", "receiver_geometry"]
            ].copy()
            comparison["_mean"] = values.loc[valid]
            best_values = comparison.groupby(
                ["dataset_name", "receiver_geometry"], dropna=False
            )["_mean"].transform("min")
            result.loc[comparison.index[comparison["_mean"].eq(best_values)], best_column] = "↓"

    group_order = {str(name): index for index, name in enumerate(dataset_order)}
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    geometry_order = {name: index for index, name in enumerate(_GEOMETRY_LABELS)}
    return (
        result.assign(
            _group_order=result["dataset_name"].map(group_order).fillna(len(group_order)),
            _geometry_order=result["receiver_geometry"].map(geometry_order).fillna(len(geometry_order)),
            _model_order=result["model_name"].map(model_order).fillna(len(model_order)),
        )
        .sort_values(
            ["_group_order", "_geometry_order", "_model_order", "model_name"],
            kind="stable",
        )
        .drop(columns=["_group_order", "_geometry_order", "_model_order"])
        .reset_index(drop=True)
    )


def write_scenario_runtime_presentation_table(
        presentation_config: Mapping[str, Any] | str | Path | None = None,
        *,
        evaluation_root: str | Path = EVALUATION_ROOT,
) -> tuple[pd.DataFrame, Path]:
    """Lê as métricas salvas, atualiza o CSV agrupado e retorna tabela e caminho."""
    if presentation_config is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "evaluation_presentation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif isinstance(presentation_config, Mapping):
        config = dict(presentation_config)
    else:
        config = json.loads(Path(presentation_config).read_text(encoding="utf-8"))

    evaluation_root = Path(evaluation_root)
    version_root = evaluation_root / str(config["version"])
    dataset_tables: list[pd.DataFrame] = []
    for dataset in config.get("datasets", ()):
        dataset_name = str(dataset["name"])
        relative_dir = dataset.get(
            "output_dir",
            config["scenario_output_dir"].format(dataset=dataset_name),
        )
        source_path = version_root / relative_dir / "per_sample_metrics.csv"
        if source_path.is_file():
            dataset_tables.append(pd.read_csv(source_path))

    per_sample_rows = (
        pd.concat(dataset_tables, ignore_index=True, sort=False)
        if dataset_tables
        else pd.DataFrame()
    )
    table = aggregate_scenario_runtime_metrics(
        per_sample_rows,
        dataset_order=[str(dataset["name"]) for dataset in config.get("datasets", ())],
    )
    summary_path = version_root / config.get(
        "runtime_summary_filename", "presentation_runtime_by_dataset.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_path, index=False)
    return table, summary_path


def evaluate_training_scenarios(
        *,
        version: str = "v6",
        model_names: Sequence[str] | None = None,
        dataset_name: str = "Marmousi",
        checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
        output_dir: str | Path | None = None,
        sample_indices: Sequence[int] | None = None,
        device: torch.device | None = None,
        performance_benchmark: Mapping[str, Any] | None = None,
) -> EvaluationOutput:
    """Avalia todos os checkpoints de cenários dos modelos selecionados."""
    selected_models = tuple(model_names) if model_names is not None else MODEL_NAMES
    checkpoint_dir = _version_root(checkpoint_root, version)
    checkpoints: list[tuple[Path, dict[str, Any]]] = []
    if checkpoint_dir.is_dir():
        for path in sorted(
            checkpoint_dir.glob(f"*_{dataset_name}_*_{EVALUATION_CHECKPOINT}.pt")
        ):
            factors = _scenario_from_path(path, None, dataset_name)
            if factors is not None and factors["model_name"] in selected_models:
                checkpoints.append((path, factors))
    if sample_indices is not None:
        selected = {int(index) for index in sample_indices}
        checkpoints = [
            (path, factors)
            for path, factors in checkpoints
            if factors["sample_id"] in selected
        ]
    models_with_checkpoints = {factors["model_name"] for _, factors in checkpoints}
    available_models = tuple(
        name for name in MODEL_NAMES
        if name in selected_models and name in models_with_checkpoints
    )
    available_models += tuple(
        name for name in selected_models
        if name in models_with_checkpoints and name not in available_models
    )
    model_rank = {name: index for index, name in enumerate(available_models)}
    checkpoints.sort(key=lambda item: (
        item[1]["noise_std_ratio"],
        item[1]["epochs"],
        item[1]["receiver_geometry"],
        model_rank[item[1]["model_name"]],
        item[1]["sample_id"],
    ))

    destination = (
        Path(output_dir)
        if output_dir is not None
        else EVALUATION_ROOT / version / f"{dataset_name}_model_scenarios"
    )
    job = EvaluationJob(
        name=f"{dataset_name}_{version}_model_scenarios",
        model_names=available_models or selected_models,
        dataset_names=(dataset_name,),
        checkpoint_dirs=(checkpoint_dir,),
        use_only_checkpoint_dirs=True,
        output_dir=destination,
    )
    output = EvaluationOutput(job=job)
    if not checkpoints:
        output.manifest.append({
            "dataset_name": dataset_name,
            "status": "missing_checkpoint",
            "checkpoint_path": str(checkpoint_dir),
            "message": "Nenhum checkpoint de cenário dos modelos selecionados foi encontrado.",
        })
        return output

    run_configs_base = {
        model_name: build_run_config(dataset_name, model_name)
        for model_name in available_models
    }
    eval_device = device or resolve_device("cuda_if_available")
    benchmark = _normalize_inference_benchmark(performance_benchmark)
    benchmark_environment = _inference_benchmark_environment(
        eval_device,
        warmup_runs=benchmark["warmup_runs"],
        timed_runs=benchmark["timed_runs"],
    )
    selected_samples = sorted({factors["sample_id"] for _, factors in checkpoints})
    target_by_sample: dict[int, np.ndarray] = {}

    for sample_id in selected_samples:
        sample_checkpoints = [
            (path, factors)
            for path, factors in checkpoints
            if factors["sample_id"] == sample_id
        ]
        for checkpoint_path, factors in sample_checkpoints:
            model_name = factors["model_name"]
            noise_job = EvaluationJob(
                name=job.name,
                model_names=(model_name,),
                dataset_names=(dataset_name,),
                noise_std_ratio=(
                    None
                    if factors["noise_std_ratio"] == 0.0
                    else factors["noise_std_ratio"]
                ),
            )
            run_config = apply_evaluation_noise(
                deepcopy(run_configs_base[model_name]),
                noise_job,
            )
            run_config["seed"] = int(run_config.get("seed", 42))
            try:
                checkpoint = load_checkpoint_low_memory(checkpoint_path)
                saved_physics = checkpoint.get("physics_config")
                if saved_physics is not None:
                    run_config["physics"] = deepcopy(saved_physics)
                torch.manual_seed(int(run_config["seed"]))
                data = prepare_data(
                    run_config,
                    torch.device("cpu"),
                    sample_index=sample_id,
                )
                target = np.asarray(data["raw_y"][0, 0]).copy()
                previous_target = target_by_sample.get(sample_id)
                if previous_target is not None and not np.array_equal(previous_target, target):
                    raise EvaluationError(
                        f"Os cenários retornaram mapas alvo diferentes para a amostra {sample_id}."
                    )
                target_by_sample[sample_id] = target
                output.true_velocity_by_sample.setdefault(dataset_name, {})[sample_id] = target

                run = EvaluationRun(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    sample_index=sample_id,
                    checkpoint_path=checkpoint_path,
                    run_name=checkpoint_path.stem.removesuffix(
                        f"_{EVALUATION_CHECKPOINT}"
                    ),
                    model_label=factors["scenario_label"],
                    noise_label=factors["noise_label"],
                )
                prediction, timing, peak_memory_mb = predict_checkpoint(
                    run,
                    checkpoint,
                    run_config,
                    data,
                    eval_device,
                    warmup_runs=benchmark["warmup_runs"],
                    timed_runs=benchmark["timed_runs"],
                    benchmark_environment=benchmark_environment,
                )
                row = score_prediction(
                    run,
                    checkpoint,
                    prediction,
                    target,
                    timing,
                    peak_memory_mb,
                    data["physics_config"],
                )
                row.update({
                    "noise_std_ratio": factors["noise_std_ratio"],
                    "epochs_target": factors["epochs"],
                    "receiver_geometry": factors["receiver_geometry"],
                })
                output.rows.append(row)
                output.manifest.append({**row, "message": ""})
                output.predictions_by_sample.setdefault(dataset_name, {}).setdefault(sample_id, {})[
                    factors["scenario_label"]
                ] = prediction
            except EvaluationError as error:
                output.manifest.append({
                    **factors,
                    "dataset_name": dataset_name,
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "model_label": factors["scenario_label"],
                    "run_name": checkpoint_path.stem.removesuffix(
                        f"_{EVALUATION_CHECKPOINT}"
                    ),
                    "status": error.status,
                    "checkpoint_path": str(checkpoint_path),
                    "message": str(error),
                })
            finally:
                if "data" in locals():
                    del data
                if "checkpoint" in locals():
                    del checkpoint
                _clear_memory()

    output.summary = summarize_metrics(output.rows)
    return output


def evaluate_mamba_scenarios(
        *,
        version: str = "v6",
        dataset_name: str = "Marmousi",
        checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
        output_dir: str | Path | None = None,
        sample_indices: Sequence[int] | None = None,
        device: torch.device | None = None,
) -> EvaluationOutput:
    """Mantém compatibilidade com a avaliação de cenários somente Mamba."""
    return evaluate_training_scenarios(
        version=version,
        model_names=("mamba",),
        dataset_name=dataset_name,
        checkpoint_root=checkpoint_root,
        output_dir=output_dir,
        sample_indices=sample_indices,
        device=device,
    )


def save_scenario_velocity_figures(
        output: EvaluationOutput,
) -> list[Path]:
    """Salva em uma figura os mapas de velocidade disponíveis de cada amostra."""
    if not output.predictions_by_sample:
        return []
    dataset_name = next(iter(output.predictions_by_sample))
    physics_config = resolve_physics_config(
        build_run_config(dataset_name, output.job.model_names[0])
    )
    figure_dir = Path(output.job.output_dir or EVALUATION_ROOT / output.job.name)
    figure_dir = figure_dir / "figures" / "model_scenarios"
    saved: list[Path] = []
    for sample_id, predictions in output.predictions_by_sample[dataset_name].items():
        if not predictions:
            continue
        path = figure_dir / f"{dataset_name}_s{sample_id:04d}_velocity_maps.png"
        plot_velocity_scenario_comparison(
            sample_id,
            output.true_velocity_by_sample[dataset_name][sample_id],
            predictions,
            vmin=float(physics_config["velocity_min"]),
            vmax=float(physics_config["velocity_max"]),
            save_path=path,
        )
        saved.append(path)
    return saved


def evaluate_scenario_dataset(
        dataset_name: str,
        *,
        version: str,
        checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
        output_dir: str | Path,
        factor_table_path: str | Path,
        presentation_table_path: str | Path,
        metric_definitions: Sequence[dict[str, Any]],
        performance_definitions: Sequence[dict[str, Any]],
        sample_indices: Sequence[int] | None = None,
        device: torch.device | None = None,
        performance_benchmark: Mapping[str, Any] | None = None,
        reuse_completed: bool = True,
) -> ScenarioEvaluationResult:
    """Executa e exporta um dataset, reusando resultados completos quando disponíveis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_device = device or resolve_device("cuda_if_available")
    benchmark = _normalize_inference_benchmark(performance_benchmark)
    benchmark_environment = _inference_benchmark_environment(
        eval_device,
        warmup_runs=benchmark["warmup_runs"],
        timed_runs=benchmark["timed_runs"],
    )
    benchmark_protocol = _inference_benchmark_protocol(benchmark)
    factor_table = build_training_factor_table(
        version=version,
        dataset_name=dataset_name,
        checkpoint_root=checkpoint_root,
    )
    factor_path = Path(factor_table_path)
    presentation_path = Path(presentation_table_path)
    per_sample_path = output_dir / "per_sample_metrics.csv"
    manifest_path = output_dir / "evaluation_manifest.csv"

    if (
        reuse_completed
        and sample_indices is None
        and not factor_table.empty
        and factor_path.is_file()
        and per_sample_path.is_file()
        and manifest_path.is_file()
    ):
        saved_factor_table = pd.read_csv(factor_path)
        current_scenario_ids = set(factor_table["scenario_id"].astype(str))
        saved_factor_table = saved_factor_table.loc[
            saved_factor_table["scenario_id"].astype(str).isin(current_scenario_ids)
        ].reset_index(drop=True)
        saved_by_id = {
            str(row["scenario_id"]): row
            for row in saved_factor_table.to_dict(orient="records")
            if row.get("scenario_id")
        }
        try:
            saved_rows = pd.read_csv(per_sample_path)
        except pd.errors.EmptyDataError:
            saved_rows = pd.DataFrame()
        complete = (
            not saved_rows.empty
            and "inference_benchmark_protocol" in saved_rows.columns
            and set(saved_rows["inference_benchmark_protocol"].dropna().astype(str))
            == {benchmark_protocol}
            and "benchmark_environment" in saved_rows.columns
            and set(saved_rows["benchmark_environment"].dropna().astype(str))
            == {benchmark_environment}
        )
        for row in factor_table.to_dict(orient="records"):
            expected_count = int(row.get("checkpoint_count", 0) or 0)
            if expected_count <= 0:
                continue
            saved = saved_by_id.get(str(row["scenario_id"]))
            if (
                saved is None
                or saved.get("checkpoint_selection") != EVALUATION_CHECKPOINT
                or int(saved.get("checkpoint_count", 0) or 0) != expected_count
                or saved.get("evaluation_status") != "calculada"
                or int(saved.get("evaluated_samples", 0) or 0) < expected_count
            ):
                complete = False
                break
        if complete:
            factor_table = saved_factor_table
            factor_table.to_csv(factor_path, index=False)
            presentation_table = scenario_presentation_table(factor_table)
            presentation_path.parent.mkdir(parents=True, exist_ok=True)
            presentation_table.to_csv(presentation_path, index=False)
            model_names = tuple(
                name for name in MODEL_NAMES
                if name in set(factor_table.get("model_name", pd.Series(dtype=str)).astype(str))
            )
            job = EvaluationJob(
                name=f"{dataset_name}_{version}_model_scenarios",
                model_names=model_names,
                dataset_names=(dataset_name,),
                checkpoint_dirs=(_version_root(checkpoint_root, version),),
                use_only_checkpoint_dirs=True,
                output_dir=output_dir,
            )
            output = EvaluationOutput(job=job)
            output.rows = pd.read_csv(per_sample_path).to_dict(orient="records")
            output.manifest = pd.read_csv(manifest_path).to_dict(orient="records")
            output.summary = summarize_metrics(output.rows)
            map_dir = output_dir / "figures" / "model_scenarios"
            map_figures = sorted(map_dir.glob(f"{dataset_name}_s*_velocity_maps.png"))
            if model_names:
                physics_config = resolve_physics_config(
                    build_run_config(dataset_name, model_names[0])
                )
                available_models = set(
                    factor_table.get("model_name", pd.Series(dtype=str)).astype(str)
                )
                plot_model_labels = {
                    name: model_label(name)
                    for name in MODEL_NAMES
                    if name in available_models
                }
                for map_figure in map_figures:
                    sample_match = re.search(r"_s(\d+)_velocity_maps$", map_figure.stem)
                    if sample_match is None:
                        continue
                    reflow_cached_velocity_scenario_figure(
                        map_figure,
                        sample_id=int(sample_match.group(1)),
                        scenario_factors=factor_table,
                        model_labels=plot_model_labels,
                        vmin=float(physics_config["velocity_min"]),
                        vmax=float(physics_config["velocity_max"]),
                    )
            charts = plot_scenario_charts(
                presentation_table,
                dataset_name=dataset_name,
                output_dir=output_dir,
                metric_definitions=metric_definitions,
                performance_definitions=performance_definitions,
            )
            return ScenarioEvaluationResult(
                output=output,
                factor_table=factor_table,
                presentation_table=presentation_table,
                factor_table_path=factor_path,
                presentation_table_path=presentation_path,
                map_figures=map_figures,
                metric_figures=charts["metrics"],
                performance_figures=charts["performance"],
                reused_existing=True,
            )

    output = evaluate_training_scenarios(
        version=version,
        dataset_name=dataset_name,
        checkpoint_root=checkpoint_root,
        output_dir=output_dir,
        sample_indices=sample_indices,
        device=eval_device,
        performance_benchmark=benchmark,
    )
    factor_table = add_scenario_evaluation_metrics(factor_table, output)
    presentation_table = scenario_presentation_table(factor_table)
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    presentation_path.parent.mkdir(parents=True, exist_ok=True)
    factor_table.to_csv(factor_path, index=False)
    presentation_table.to_csv(presentation_path, index=False)

    save_evaluation_outputs(output)
    save_presentation_tables(output)
    map_figures = save_scenario_velocity_figures(output)
    charts = plot_scenario_charts(
        presentation_table,
        dataset_name=dataset_name,
        output_dir=output_dir,
        metric_definitions=metric_definitions,
        performance_definitions=performance_definitions,
    )
    return ScenarioEvaluationResult(
        output=output,
        factor_table=factor_table,
        presentation_table=presentation_table,
        factor_table_path=factor_path,
        presentation_table_path=presentation_path,
        map_figures=map_figures,
        metric_figures=charts["metrics"],
        performance_figures=charts["performance"],
        reused_existing=False,
    )
