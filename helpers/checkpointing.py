from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

from config.config import DEFAULT_CHECKPOINT_ROOT, resolve_physics_config
from data.files import resolve_dataset_files

class CheckpointArchitectureError(ValueError):
    """Indica que o checkpoint não contém uma arquitetura salva compatível."""


_ARCHITECTURE_KEYS = (
    "model_name", "dataset_name", "in_channels", "n_blocks_encoder",
    "n_blocks_decoder", "base_channels", "final_spatial_shape",
    "final_out_channels", "decoder_initial_shape", "decoder_initial_channels",
    "final_size_encoder", "encoder_base_channels", "encoder_pool_shape",
    "batch_size", "m_min", "m_max", "model_batch_size", "model_output_min",
    "model_output_max", "mamba_base_channels", "nt", "nr", "use_mamba_encoder",
    "use_attention",
    "use_mamba_bottleneck",
    "use_mamba_decoder", "use_skip_connections", "mamba_d_conv",
    "mamba_d_state", "mamba_expand", "mamba_dropout",
    "mamba_conv_kernel_size", "use_pos_embed", "dim1", "dim2",
    "dim3", "dim4", "dim5", "sample_spatial",
    "pgann_encoder_channel_scale",
    "pgann_decoder_channel_scale",
    "pgann_attention_latent_kernel_size",
    "pgann_attention_feature_kernel_size",
    "pgann_attention_score_kernel_size",
    "pgann_attention_reduction",
)


def checkpoint_architecture(
        model: torch.nn.Module,
        hyperparameters: dict[str, Any] | None,
        *,
        state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Armazena os valores do construtor e as dimensões dos tensores de um modelo."""
    saved_hyperparameters = hyperparameters or {}
    if state_dict is None:
        state_dict = model.state_dict()
    constructor_parameters = {
        key: copy.deepcopy(saved_hyperparameters[key])
        for key in _ARCHITECTURE_KEYS
        if key in saved_hyperparameters
    }
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "constructor_parameters": constructor_parameters,
        "state_dict_shapes": {
            name: list(tensor.shape)
            for name, tensor in state_dict.items()
            if torch.is_tensor(tensor)
        },
    }


def checkpoint_hyperparameters(
        checkpoint: dict[str, Any],
        model_name: str,
) -> dict[str, Any]:
    """Retorna os argumentos do construtor salvos sem consultar a configuração atual."""
    architecture = checkpoint.get("architecture") or {}
    parameters = architecture.get("constructor_parameters")
    if not isinstance(parameters, dict):
        raise CheckpointArchitectureError(
            "O checkpoint não possui arquitetura salva; converta-o antes da avaliação."
        )
    parameters = dict(parameters)
    parameters["model_name"] = model_name
    return parameters


def training_signature(model_name: str, config: dict) -> str | None:
    """Invalida execuções de Marmousi quando o modelo, a física ou as observações mudam."""
    if config.get("dataset_name") != "Marmousi":
        return None
    training = dict(config["training"])
    if model_name == "pgnn_mardan_2024":
        # Mantenha as assinaturas estáveis durante a transferência dos padrões
        # da arquitetura PGNN do executor para o arquivo de configuração.
        if "model_batch_size" in training:
            training["pgnn_mardan_2024_batch_size"] = training.pop("model_batch_size")
        if training.get("use_attention") is False:
            training.pop("use_attention")
        if training.get("model_output_min") == 0.0 and training.get("model_output_max") == 1.0:
            training.pop("model_output_min")
            training.pop("model_output_max")
        decoder_shape = training.get("decoder_initial_shape")
        if (
            decoder_shape is not None
            and training.get("final_size_encoder")
            == int(decoder_shape[0]) * int(decoder_shape[1])
        ):
            training.pop("final_size_encoder")
    for key in (
        "report_freq",
        "verbose",
        "save_training_log",
        "save_checkpoints",
        "measure_runtime",
    ):
        training.pop(key, None)
    files = resolve_dataset_files(config["datasets_root"], config["dataset_name"])
    payload = {
        "version": 1,
        "model": model_name,
        "training": training,
        "physics": resolve_physics_config(config),
        "seed": config.get("seed", 42),
        "data": [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def training_signature_candidates(model_name: str, config: dict) -> frozenset[str] | None:
    """Retorna assinaturas equivalentes de treino, incluindo métricas de seleção antigas."""
    config_variants = [config]
    metric_values = (None, "total_loss")
    if config.get("dataset_name") == "Marmousi":
        for metric in metric_values:
            variant = copy.deepcopy(config)
            training = variant.setdefault("training", {})
            if metric is None:
                training.pop("checkpoint_metric", None)
            else:
                training["checkpoint_metric"] = metric
            config_variants.append(variant)

    candidates: set[str] = set()
    for variant in config_variants:
        signature = training_signature(model_name, variant)
        if signature is not None:
            candidates.add(signature)

        training = variant.get("training", {})
        filter_type = training.get("loss_filter_type")
        if variant.get("dataset_name") != "Marmousi" or filter_type not in (None, "butterworth"):
            continue
        legacy_config = copy.deepcopy(variant)
        legacy_training = legacy_config.setdefault("training", {})
        if filter_type is None:
            legacy_training["loss_filter_type"] = "butterworth"
        else:
            legacy_training.pop("loss_filter_type", None)
        legacy_signature = training_signature(model_name, legacy_config)
        if legacy_signature is not None:
            candidates.add(legacy_signature)

    return frozenset(candidates) or None


def _to_cpu_and_clone(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: _to_cpu_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_cpu_and_clone(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_cpu_and_clone(v) for v in obj)
    else:
        return copy.deepcopy(obj)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    **kwargs,
) -> dict[str, Any]:
    """Monta o conteúdo comum usado em salvamentos diretos e checkpoints best/last."""
    state_dict = model.state_dict()
    if "architecture" not in kwargs:
        kwargs["architecture"] = checkpoint_architecture(
            model, kwargs.get("hyperparameters"), state_dict=state_dict,
        )
    return {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        **kwargs,
    }


def _write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    **kwargs,
) -> Path:
    return _write_checkpoint(path, _checkpoint_payload(model, optimizer, **kwargs))


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
) -> dict[str, Any]:
    """Lê um checkpoint confiável do projeto, incluindo otimizador e metadados da execução."""
    options = {"map_location": map_location, "weights_only": False}
    if mmap:
        options["mmap"] = True
    try:
        return torch.load(path, **options)
    except TypeError:
        # Compatibilidade com versões do PyTorch anteriores à opção mmap.
        if not mmap:
            raise
        options.pop("mmap")
        return torch.load(path, **options)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_checkpoint_metadata(
    path: str | Path, *, expected_signature: str | Collection[str] | None = None,
) -> dict[str, Any] | None:
    try:
        try:
            checkpoint = read_checkpoint(path, mmap=True)
        except RuntimeError:
            # Checkpoints saved in the legacy, non-zip format cannot be memory-mapped.
            checkpoint = read_checkpoint(path)
    except (OSError, RuntimeError, ValueError, EOFError):
        return None
    if expected_signature is not None:
        expected_signatures = (
            {expected_signature}
            if isinstance(expected_signature, str)
            else set(expected_signature)
        )
        if checkpoint.get("training_signature") not in expected_signatures:
            return None
    return checkpoint


def completed_checkpoint_path(
    checkpoint_prefix: str,
    expected_epoch: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
    expected_signature: str | Collection[str] | None = None,
) -> Path | None:
    path = Path(checkpoint_dir) / f"{checkpoint_prefix}_last.pt"
    checkpoint = load_checkpoint_metadata(path, expected_signature=expected_signature)
    if checkpoint is not None and int(checkpoint.get("epoch", -1)) >= int(expected_epoch):
        return path
    return None


def resumable_checkpoint_path(
    checkpoint_prefix: str,
    expected_epoch: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
    expected_signature: str | Collection[str] | None = None,
) -> Path | None:
    path = Path(checkpoint_dir) / f"{checkpoint_prefix}_last.pt"
    checkpoint = load_checkpoint_metadata(path, expected_signature=expected_signature)
    if checkpoint is None:
        return None

    epoch = int(checkpoint.get("epoch", -1))
    if 0 <= epoch < int(expected_epoch):
        return path
    return None


class CheckpointManager:
    def __init__(
        self,
        checkpoint_prefix: str,
        save_enabled: bool = True,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
        save_freq: int = 1,
        load_existing_best: bool = True,
        expected_signature: str | Collection[str] | None = None,
    ):
        self.checkpoint_prefix = checkpoint_prefix
        self.save_enabled = save_enabled
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = max(1, int(save_freq))
        self.best_checkpoint_value = float("inf")
        self.best_checkpoint_metric = "total_loss"
        self.best_total_loss = float("inf")
        self.best_model_loss = float("inf")
        self.best_epoch = 0
        self.best_model_buffer = None

        best_checkpoint = (
            load_checkpoint_metadata(
                self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt",
                expected_signature=expected_signature,
            )
            if load_existing_best and save_enabled else None
        )
        if (
            best_checkpoint is not None
            and best_checkpoint.get("best_checkpoint_metric") == "total_loss"
        ):
            self.best_total_loss = float(best_checkpoint.get("total_loss", self.best_total_loss))
            self.best_model_loss = float(best_checkpoint.get("model_loss", self.best_model_loss))
            saved_total_loss = self._finite_loss(best_checkpoint.get("total_loss"))
            self.best_checkpoint_value = (
                saved_total_loss if saved_total_loss is not None else float("inf")
            )
            self.best_epoch = int(best_checkpoint.get("epoch", self.best_epoch))

    @staticmethod
    def _finite_loss(value: Any) -> float | None:
        if value is None:
            return None
        try:
            loss = float(value)
        except (TypeError, ValueError):
            return None
        return loss if math.isfinite(loss) else None

    def _checkpoint_score(self, total_loss: Any) -> float:
        total_loss_value = self._finite_loss(total_loss)
        return total_loss_value if total_loss_value is not None else float("inf")

    def update(
        self,
        total_loss: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        final: bool = False,
        **kwargs,
    ) -> bool:
        model_loss = kwargs.get("model_loss")
        checkpoint_score = self._checkpoint_score(total_loss)
        is_best = checkpoint_score < self.best_checkpoint_value
        if is_best:
            self.best_checkpoint_value = checkpoint_score
            self.best_checkpoint_metric = "total_loss"
            self.best_total_loss = total_loss
            model_loss_value = self._finite_loss(model_loss)
            self.best_model_loss = model_loss_value if model_loss_value is not None else float("inf")
            self.best_epoch = epoch

        if not self.save_enabled:
            return is_best

        # Mantenha o melhor modelo em memória.
        if is_best:
            self.best_model_buffer = _to_cpu_and_clone(_checkpoint_payload(
                model,
                optimizer,
                **{
                    "epoch": epoch,
                    "total_loss": total_loss,
                    "best_checkpoint_metric": self.best_checkpoint_metric,
                    "best_checkpoint_value": checkpoint_score,
                    **kwargs,
                },
            ))

        should_save = final or epoch == 1 or epoch % self.save_freq == 0
        if should_save:
            save_checkpoint(
                self.checkpoint_dir / f"{self.checkpoint_prefix}_last.pt",
                model,
                optimizer,
                epoch=epoch,
                total_loss=total_loss,
                **kwargs,
            )
            if self.best_model_buffer is not None:
                best_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
                _write_checkpoint(best_path, self.best_model_buffer)
                self.best_model_buffer = None

        return is_best
