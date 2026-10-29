from __future__ import annotations

import gc
import traceback
from typing import Any, Sequence

import torch

from config.config import DEFAULT_CHECKPOINT_ROOT, TRAINING_LOGS_ROOT, resolve_physics_config
from data.dataset import configured_sample_indices, prepare_data
from data.seismic_noise import noise_condition_label, noise_run_suffix
from helpers.checkpointing import (
    completed_checkpoint_path,
    training_signature_candidates,
)
from network.models import build_model
from network.pipeline import train_model
from utils import seed_everything


def release_memory(device: torch.device) -> None:
    """Libera a memória do Python e as alocações CUDA em cache entre execuções independentes."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def training_sample_checkpoint_prefix(
        model_name: str,
        run_config: dict[str, Any],
        sample_index: int,
) -> str:
    """Monta o prefixo usado para localizar o checkpoint de uma amostra."""
    training = run_config.get("training", {})
    run_name = (
        f"{model_name}_{run_config['dataset_name']}"
        f"{noise_run_suffix(training.get('seismic_input_noise_std_ratio'), training.get('seismic_observed_noise_std_ratio'))}"
    )
    if run_config.get("run_label"):
        run_name = f"{run_name}_{run_config['run_label']}"
    return f"{run_name}_s{int(sample_index):04d}"


def pending_training_sample_indices(
        model_name: str,
        run_config: dict[str, Any],
        sample_indices: Sequence[int],
        *,
        force: bool = False,
) -> list[int]:
    """Retorna somente amostras sem checkpoint concluído para esta configuração."""
    indices = [int(index) for index in sample_indices]
    if force:
        return indices

    epochs = int(run_config["training"]["epochs"])
    checkpoint_dir = run_config.get("checkpointing", {}).get("dir", DEFAULT_CHECKPOINT_ROOT)
    expected_signature = training_signature_candidates(model_name, run_config)
    return [
        sample_index
        for sample_index in indices
        if completed_checkpoint_path(
            training_sample_checkpoint_prefix(model_name, run_config, sample_index),
            epochs,
            checkpoint_dir=checkpoint_dir,
            expected_signature=expected_signature,
        ) is None
    ]


def build_hyperparameters(
        model_name: str,
        run_config: dict[str, Any],
        data: dict[str, Any],
        sample_index: int,
) -> dict[str, Any]:
    """Monta os argumentos da fábrica de modelos para uma amostra de treinamento preparada."""
    training = run_config["training"]
    model_shape = tuple(int(value) for value in data["model_shape"])
    n_blocks_encoder = int(training["n_blocks_encoder"])
    physics_config = data.get("physics_config") or resolve_physics_config(run_config)
    hyperparameters = dict(training)
    hyperparameters.update({
        "model_name": model_name,
        "dataset_name": run_config["dataset_name"],
        "sample_index": sample_index,
        "noise_condition": noise_condition_label(
            training.get("seismic_input_noise_std_ratio"),
            training.get("seismic_observed_noise_std_ratio"),
        ),
        "seismic_input_noise_std_ratio": training.get("seismic_input_noise_std_ratio"),
        "seismic_observed_noise_std_ratio": training.get("seismic_observed_noise_std_ratio"),
        "in_channels": int(physics_config["source_count"]),
        "n_blocks_encoder": n_blocks_encoder,
        "n_blocks_decoder": int(training["n_blocks_decoder"]),
        "final_spatial_shape": model_shape,
        "nt": int(data["raw_x"].shape[-2]),
        "nr": int(data["raw_x"].shape[-1]),
        "final_out_channels": int(training.get("final_out_channels", 1)),
    })
    for key in ("decoder_initial_shape", "encoder_pool_shape"):
        if key in hyperparameters:
            hyperparameters[key] = tuple(int(value) for value in hyperparameters[key])

    return hyperparameters


def run_training_sample(
        model_name: str,
        run_config: dict[str, Any],
        device: torch.device,
        sample_index: int,
        *,
        force: bool = False,
        resume: bool = True,
        progress_reporter: Any | None = None,
) -> dict[str, Any]:
    """Executa o fluxo compartilhado de preparação, construção e treinamento para uma amostra de dataset."""
    sample_index = int(sample_index)
    dataset_name = str(run_config["dataset_name"])
    epochs = int(run_config["training"]["epochs"])
    training = run_config.get("training", {})
    run_name = training_sample_checkpoint_prefix(model_name, run_config, sample_index)
    noise_label = noise_run_suffix(
        training.get("seismic_input_noise_std_ratio"),
        training.get("seismic_observed_noise_std_ratio"),
    ).lstrip("_").replace("noise_", "noise", 1) or "clean"
    results_version = str(run_config.get("results_version", "v1"))
    log_path = TRAINING_LOGS_ROOT / results_version / noise_label / f"{run_name}_training_log.csv"
    checkpoint_dir = run_config.get("checkpointing", {}).get("dir", DEFAULT_CHECKPOINT_ROOT)

    if progress_reporter is not None:
        progress_reporter.start_task(epochs)

    completed_checkpoint = completed_checkpoint_path(
        run_name,
        epochs,
        checkpoint_dir=checkpoint_dir,
        expected_signature=training_signature_candidates(model_name, run_config),
    )
    if completed_checkpoint is not None and not force:
        print(f"{run_name} já foi concluído; ignorando ({completed_checkpoint})")
        if progress_reporter is not None:
            progress_reporter.finish_task(skipped=True)
        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "noise_label": noise_condition_label(
                training.get("seismic_input_noise_std_ratio"),
                training.get("seismic_observed_noise_std_ratio"),
            ),
            "sample_index": sample_index,
            "epoch": epochs,
            "checkpoint_path": str(completed_checkpoint),
            "skipped": True,
        }

    release_memory(device)
    data = None
    model = None
    try:
        seed_everything(run_config.get("seed", 42))
        data = prepare_data(run_config, device, sample_index=sample_index)
        hyperparameters = build_hyperparameters(model_name, run_config, data, sample_index)
        model = build_model(model_name, hyperparameters, device)

        print(run_name)
        summary = train_model(
            run_name,
            model,
            hyperparameters,
            run_config,
            device,
            checkpoint_prefix=run_name,
            log_path=log_path,
            data=data,
            epoch_callback=(progress_reporter.on_epoch if progress_reporter is not None else None),
            resume=resume and not force,
        )
        summary["dataset_name"] = dataset_name
        summary["noise_label"] = noise_condition_label(
            training.get("seismic_input_noise_std_ratio"),
            training.get("seismic_observed_noise_std_ratio"),
        )
        summary["sample_index"] = sample_index
        summary["skipped"] = False
        return summary
    except BaseException as exc:
        # Tracebacks keep failed training frames and their tensors alive.
        errors = [exc]
        seen = set()
        while errors:
            error = errors.pop()
            if id(error) not in seen:
                seen.add(id(error))
                traceback.clear_frames(error.__traceback__)
                errors.extend(
                    cause for cause in (error.__cause__, error.__context__)
                    if cause is not None
                )
        raise
    finally:
        del model, data
        release_memory(device)
        if progress_reporter is not None:
            progress_reporter.finish_task()


def run_training_samples(
        model_name: str,
        run_config: dict[str, Any],
        device: torch.device,
        *,
        sample_indices: Sequence[int] | None = None,
        force: bool = False,
        resume: bool = True,
        progress_reporter: Any | None = None,
) -> list[dict[str, Any]]:
    """Executa uma por vez todas as amostras selecionadas pelo fluxo compartilhado."""
    indices = (
        configured_sample_indices(run_config)
        if sample_indices is None
        else [int(value) for value in sample_indices]
    )
    summaries: list[dict[str, Any]] = []
    try:
        for sample_index in indices:
            summaries.append(
                run_training_sample(
                    model_name,
                    run_config,
                    device,
                    sample_index,
                    force=force,
                    resume=resume,
                    progress_reporter=progress_reporter,
                )
            )
        return summaries
    finally:
        release_memory(device)
