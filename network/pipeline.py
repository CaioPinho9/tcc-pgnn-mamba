from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import prepare_data, prepare_training_sample
from network.train import train_one_epoch
from config.config import DEFAULT_CHECKPOINT_ROOT, resolve_physics_config
from data.paths import ensure_output_dirs
from config.training import TrainingOptions
from physics.forward import build_forward_operator
from transformation.min_max_normalizer import MinMaxNormalizer

from helpers.checkpointing import (
    CheckpointManager,
    count_trainable_parameters,
    load_checkpoint,
    resumable_checkpoint_path,
    training_signature,
    training_signature_candidates,
)
from helpers.logger import TrainingLogger
from helpers.runtime import (
    gpu_count,
    measure_inference_time,
    peak_gpu_memory_mb,
    reset_peak_memory,
    synchronize_if_cuda,
)


@torch.no_grad()
def evaluate_model_l1_structural_loss(
        model: torch.nn.Module,
        training_sample: dict[str, torch.Tensor],
        velocity_normalizer: MinMaxNormalizer,
) -> float:
    model.eval()

    cnn_input = training_sample["cnn_input"]
    target = training_sample["target_velocity"]
    predicted_velocity = velocity_normalizer.denormalize(model(cnn_input))
    total_abs_error = torch.sum(torch.abs(predicted_velocity - target)).item()
    total_values = target.numel()

    return total_abs_error / max(total_values, 1)


def train_model(
        model_name: str,
        model: torch.nn.Module,
        hyperparameters: dict[str, Any],
        config: dict[str, Any],
        device: torch.device,
        *,
        checkpoint_prefix: str,
        log_path: str | Path | None,
        data: dict[str, Any] | None = None,
        epoch_callback=None,
        resume: bool = False,
) -> dict[str, Any]:
    # Preparando modelo
    ensure_output_dirs()
    if data is None:
        # A preparação entrega os traços sísmicos, a referência de velocidade e a geometria física da aquisição.
        data = prepare_data(config, device)

    physics_config = data.get("physics_config") or resolve_physics_config(config)
    velocity_normalizer = MinMaxNormalizer(target_min=0.0, target_max=1.0).fit(
        torch.tensor(
            [
                float(physics_config["velocity_min"]),
                float(physics_config["velocity_max"]),
            ],
            device=device,
        )
    )

    source_indices = data.get("source_indices")
    if source_indices is None:
        raise ValueError(
            "A configuração física deve declarar a geometria de aquisição "
            "(receiver_geometry) antes do treinamento;"
        )
    # O operador usa a geometria para simular, a partir da velocidade prevista, os sismogramas comparados na perda.
    forward_operator = build_forward_operator(
        source_indices=source_indices,
        receiver_count=data["receiver_count"],
        time_step_count=data["time_step_count"],
        receiver_indices=data.get("receiver_indices"),
        checkpoint_time_steps=config["training"].get(
            "checkpoint_time_steps",
            physics_config.get("checkpoint_time_steps"),
        ),
        config=physics_config,
    ).to(device)

    # A mesma amostra fornece a entrada da rede e os sismogramas observados.
    training_sample = prepare_training_sample(data, device)

    model = model.to(device)
    training = config["training"]
    epochs = training["epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("adam_betas", (0.9, 0.99))),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training.get("lr_milestone", 100)),
        gamma=float(training.get("lr_gamma", 0.5)),
    )
    num_parameters = count_trainable_parameters(model)

    options = TrainingOptions.from_config(config, log_path)
    checkpoint_dir = config.get("checkpointing", {}).get("dir", DEFAULT_CHECKPOINT_ROOT)
    frequency_hz = float(physics_config["frequency_hz"])

    signature = training_signature(hyperparameters.get("model_name", model_name), config)
    signature_candidates = training_signature_candidates(
        hyperparameters.get("model_name", model_name), config
    )

    start_epoch = 1
    resume_checkpoint_path = None
    if resume and options.save_checkpoints:
        resume_checkpoint_path = resumable_checkpoint_path(
            checkpoint_prefix,
            epochs,
            checkpoint_dir=checkpoint_dir,
            expected_signature=signature_candidates,
        )
        if resume_checkpoint_path is not None:
            checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, map_location="cpu")
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            else:
                scheduler.last_epoch = checkpoint_epoch
                scheduler._last_lr = [float(group["lr"]) for group in optimizer.param_groups]
            del checkpoint, scheduler_state
            start_epoch = checkpoint_epoch + 1
            print(f"{model_name} retomando de {resume_checkpoint_path} na época {start_epoch}/{epochs}")

    checkpoint_manager = CheckpointManager(
        checkpoint_prefix=checkpoint_prefix,
        save_enabled=options.save_checkpoints,
        checkpoint_dir=checkpoint_dir,
        save_freq=config.get("checkpointing", {}).get("save_freq", options.report_freq),
        load_existing_best=resume_checkpoint_path is not None,
        expected_signature=signature_candidates,
    )

    logger = TrainingLogger(
        log_path=log_path,
        model_name=model_name,
        total_epochs=epochs,
        start_epoch=start_epoch,
        report_freq=options.report_freq,
        enabled=options.save_training_log,
        verbose=options.verbose,
        reset_log=resume_checkpoint_path is None,
        resume_epoch=(start_epoch - 1) if resume_checkpoint_path is not None else None,
    )

    try:
        total_train_time = 0.0
        total_neural_network_time = 0.0
        total_physics_solver_time = 0.0
        peak_memory_overall = 0.0
        epoch_times: list[float] = []
        loss_filter_config = {
            "sample_rate_hz": training.get("loss_filter_sample_rate_hz"),
            "order": int(training.get("loss_filter_order", 3)),
            "padding": int(training.get("loss_filter_padding", 512)),
        }
        # Loop de Treinamento
        for epoch in range(start_epoch, epochs + 1):
            reset_peak_memory(device)
            synchronize_if_cuda(device)
            epoch_start = time.perf_counter()
            epoch_timings: dict[str, float] = {}

            # A época ajusta a rede para aproximar os sismogramas simulados dos observados.
            data_loss = train_one_epoch(
                model=model,
                forward_operator=forward_operator,
                optimizer=optimizer,
                cnn_input=training_sample["cnn_input"],
                observed_seismic=training_sample["observed_seismic"],
                frequency_hz=frequency_hz,
                time_spacing_s=physics_config["time_spacing_s"],
                gradient_clip_norm=config["training"].get("gradient_clip_norm"),
                seismic_normalizer=data["normalizer"],
                velocity_normalizer=velocity_normalizer,
                timings=epoch_timings,
                source_mini_batches=training.get("source_mini_batches"),
                loss_frequency_cutoffs_hz=training.get("loss_frequency_cutoffs_hz"),
                loss_filter_config=loss_filter_config,
            )
            scheduler.step()

            # Log
            synchronize_if_cuda(device)
            epoch_time = time.perf_counter() - epoch_start
            total_train_time += epoch_time
            neural_network_time = epoch_timings.get("neural_network_time_seconds", 0.0)
            physics_solver_time = epoch_timings.get("physics_solver_time_seconds", 0.0)
            total_neural_network_time += neural_network_time
            total_physics_solver_time += physics_solver_time
            epoch_times.append(epoch_time)
            peak_memory = peak_gpu_memory_mb(device)
            peak_memory_overall = max(peak_memory_overall, peak_memory)
            gpu_hours = (total_train_time / 3600.0) * gpu_count(device)
            total_loss = float(data_loss)
            learning_rate = float(optimizer.param_groups[0]["lr"])

            # A referência de velocidade mede o erro estrutural, mas não dirige o gradiente.
            model_loss = evaluate_model_l1_structural_loss(
                model,
                training_sample,
                velocity_normalizer,
            )

            if epoch == epochs and options.measure_runtime:
                inference = measure_inference_time(model, data["loader"], device)
            else:
                inference = {
                    "mean_inference_time_ms": "",
                    "std_inference_time_ms": "",
                    "num_inference_samples": "",
                }

            checkpoint_manager.update(
                total_loss=total_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                final=epoch == epochs,
                data_loss=data_loss,
                model_loss=model_loss,
                model_name=model_name,
                hyperparameters=hyperparameters,
                normalizer_state=data["normalizer"].get_state(),
                velocity_normalizer_state=velocity_normalizer.get_state(),
                scheduler_state_dict=scheduler.state_dict(),
                training_signature=signature,
                physics_config=physics_config,
            )

            log_row = {
                "model_name": model_name,
                "epoch": epoch,
                "data_loss": data_loss,
                "model_loss": model_loss,
                "total_loss": total_loss,
                "num_parameters": num_parameters,
                "neural_network_time_seconds": neural_network_time,
                "physics_solver_time_seconds": physics_solver_time,
                "total_epoch_time_seconds": epoch_time,
                "epoch_time_seconds": epoch_time,
                "total_train_time_seconds": total_train_time,
                "peak_gpu_memory_mb": peak_memory,
                "gpu_hours": gpu_hours,
                "learning_rate": learning_rate,
                "frequency_hz": frequency_hz,
                "batch_size": options.batch_size,
                "seed": options.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mean_inference_time_ms": inference["mean_inference_time_ms"],
                "std_inference_time_ms": inference["std_inference_time_ms"],
                "num_inference_samples": inference["num_inference_samples"],
            }
            if epoch_callback is not None:
                epoch_callback(log_row)
            logger.log_epoch(log_row)

        summary_row = {
            "model_name": model_name,
            "num_parameters": num_parameters,
            "total_neural_network_time_seconds": total_neural_network_time,
            "total_physics_solver_time_seconds": total_physics_solver_time,
            "total_train_time_seconds": total_train_time,
            "mean_neural_network_time_seconds": (
                total_neural_network_time / max(1, len(epoch_times))
            ),
            "mean_physics_solver_time_seconds": (
                total_physics_solver_time / max(1, len(epoch_times))
            ),
            "mean_epoch_time_seconds": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "peak_gpu_memory_mb": peak_memory_overall,
            "gpu_hours": (total_train_time / 3600.0) * gpu_count(device),
            "mean_inference_time_ms": inference["mean_inference_time_ms"],
            "std_inference_time_ms": inference["std_inference_time_ms"],
            "num_inference_samples": inference["num_inference_samples"],
            "best_total_loss": checkpoint_manager.best_total_loss,
            "best_model_loss": checkpoint_manager.best_model_loss,
            "best_checkpoint_metric": checkpoint_manager.best_checkpoint_metric,
            "best_epoch": checkpoint_manager.best_epoch,
            "final_data_loss": data_loss,
            "final_model_loss": model_loss,
            "checkpoint_path": str(checkpoint_manager.checkpoint_dir / f"{checkpoint_prefix}_best.pt") if options.save_checkpoints else None,
        }
        return summary_row
    finally:
        logger.close()
