import time
from math import isfinite
from typing import Sequence

import torch
import torch.nn.functional as F

from transformation.normalizer import Normalizer
from physics.filter import lowpass_seismic
from helpers.runtime import synchronize_if_cuda


def _add_timing(timings: dict[str, float] | None, key: str, elapsed: float) -> None:
    """Acumula um valor de tempo sem alterar o valor retornado pela API de treinamento."""
    if timings is not None:
        timings[key] = float(timings.get(key, 0.0)) + float(elapsed)


def compute_batch_source_loss(
        model,
        forward_operator,
        cnn_input,
        observed_seismic,
        frequency_hz: float,
        time_spacing_s: float,
        seismic_normalizer: Normalizer = None,
        velocity_normalizer: Normalizer = None,
        timings: dict[str, float] | None = None,
        source_selection=None,
        loss_filter_config: dict | None = None,
):
    """Calcula a perda sísmica de um lote, opcionalmente usando um grupo de fontes.

    A rede estima a velocidade a partir de ``cnn_input``; o operador direto
    simula os sismogramas dessa velocidade, que são comparados aos observados
    após a filtragem e, quando configurada, a normalização. ``source_selection``
    seleciona as mesmas fontes na simulação e nos dados observados. Os tempos
    de rede e de física são acumulados em ``timings`` quando esse dicionário é
    fornecido.

    Retorna a soma das perdas L1 dividida pelo número de fontes selecionadas.
    """
    # Os kernels CUDA são assíncronos; sincronize cada seção medida para
    # capturar o tempo real de execução, em vez do tempo de lançamento.
    synchronize_if_cuda(cnn_input.device)
    network_start = time.perf_counter()
    # A rede converte os sismogramas de entrada em uma imagem de velocidade prevista.
    predicted_velocity = model(cnn_input)  # [B, 1, nz, nx], normalizado para [0, 1]
    synchronize_if_cuda(cnn_input.device)
    _add_timing(
        timings,
        "neural_network_time_seconds",
        time.perf_counter() - network_start,
    )

    if velocity_normalizer is not None:
        predicted_velocity = velocity_normalizer.denormalize(predicted_velocity)

    synchronize_if_cuda(predicted_velocity.device)
    physics_start = time.perf_counter()
    # O solver gera sismogramas a partir da velocidade prevista, para compará-los com os observados.
    if source_selection is None:
        predicted_seismic = forward_operator(predicted_velocity)
    else:
        predicted_seismic = forward_operator(
            predicted_velocity,
            source_selection=source_selection,
        )
        observed_seismic = observed_seismic.index_select(1, source_selection)
    synchronize_if_cuda(predicted_velocity.device)
    _add_timing(
        timings,
        "physics_solver_time_seconds",
        time.perf_counter() - physics_start,
    )

    loss_filter_config = loss_filter_config or {}
    predicted_seismic = lowpass_seismic(
        predicted_seismic,
        frequency_hz,
        time_spacing_s,
        **loss_filter_config,
    )
    observed_seismic = lowpass_seismic(
        observed_seismic,
        frequency_hz,
        time_spacing_s,
        **loss_filter_config,
    )

    if seismic_normalizer is not None:
        predicted_seismic = seismic_normalizer.normalize(predicted_seismic)
        observed_seismic = seismic_normalizer.normalize(observed_seismic)

    total_sources = predicted_seismic.shape[1]

    loss_data = F.l1_loss(
        predicted_seismic,
        observed_seismic,
        reduction="sum",
    ) / total_sources

    return loss_data


def train_one_epoch(
        model,
        forward_operator,
        optimizer,
        cnn_input,
        observed_seismic,
        frequency_hz: float,
        time_spacing_s: float,
        gradient_clip_norm=None,
        seismic_normalizer: Normalizer = None,
        velocity_normalizer: Normalizer = None,
        timings: dict[str, float] | None = None,
        source_mini_batches: int | None = None,
        loss_frequency_cutoffs_hz: Sequence[float] | None = None,
        loss_filter_config: dict | None = None,
):
    """Treina a rede por uma época e retorna a perda sísmica média ponderada.

    Cada combinação de grupo de fontes e frequência de corte calcula uma perda,
    zera os gradientes, executa retropropagação e atualiza o otimizador. Quando
    ``source_mini_batches`` é definido, as fontes são repartidas em grupos;
    quando ``loss_frequency_cutoffs_hz`` é definido, cada grupo é treinado em
    cada frequência informada. A perda retornada pondera cada atualização pela
    fração de fontes usada e pela quantidade de frequências.
    """
    model.train()
    forward_operator.train(False)
    source_count = observed_seismic.shape[1]
    if source_mini_batches is not None and source_mini_batches <= 0:
        raise ValueError("source_mini_batches deve ser positivo")

    if loss_frequency_cutoffs_hz is None:
        loss_frequencies = (float(frequency_hz),)
    else:
        if isinstance(loss_frequency_cutoffs_hz, (str, bytes)):
            raise ValueError("loss_frequency_cutoffs_hz deve ser uma sequência numérica")
        loss_frequencies = tuple(float(value) for value in loss_frequency_cutoffs_hz)
        if not loss_frequencies:
            raise ValueError("loss_frequency_cutoffs_hz não pode estar vazia")
        if any(not isfinite(value) or value <= 0 for value in loss_frequencies):
            raise ValueError("Cada corte em loss_frequency_cutoffs_hz deve ser positivo e finito")

    selections = [None]
    if source_mini_batches is not None:
        selections = [
            torch.arange(group, source_count, source_mini_batches, device=cnn_input.device)
            for group in range(source_mini_batches)
        ]
        selections = [selection for selection in selections if selection.numel() > 0]

    weighted_loss = 0.0
    for selection in selections:
        batch_sources = source_count if selection is None else selection.numel()
        for loss_frequency_hz in loss_frequencies:
            optimizer.zero_grad(set_to_none=True)
            loss_data = compute_batch_source_loss(
                model,
                forward_operator,
                cnn_input,
                observed_seismic,
                frequency_hz=loss_frequency_hz,
                time_spacing_s=time_spacing_s,
                seismic_normalizer=seismic_normalizer,
                velocity_normalizer=velocity_normalizer,
                timings=timings,
                source_selection=selection,
                loss_filter_config=loss_filter_config,
            )
            if not torch.isfinite(loss_data):
                raise FloatingPointError("Perda sísmica não finita; verifique a configuração da fonte e do solver")
            synchronize_if_cuda(cnn_input.device)
            network_update_start = time.perf_counter()
            loss_data.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            synchronize_if_cuda(cnn_input.device)
            _add_timing(
                timings,
                "neural_network_time_seconds",
                time.perf_counter() - network_update_start,
            )
            weighted_loss += (
                loss_data.item()
                * batch_sources
                / source_count
                / len(loss_frequencies)
            )
            del loss_data
    return weighted_loss


@torch.no_grad()
def predict_velocity(model, dataloader, device, velocity_normalizer: Normalizer = None):
    model.eval()
    cnn_input = next(iter(dataloader))["cnn_input"].to(device)
    predicted_velocity = model(cnn_input)
    if velocity_normalizer is not None:
        predicted_velocity = velocity_normalizer.denormalize(predicted_velocity)
    return predicted_velocity


@torch.no_grad()
def predict_seismic(model, forward_operator, dataloader, device, velocity_normalizer: Normalizer = None):
    model.eval()
    batch = next(iter(dataloader))
    cnn_input = batch["cnn_input"].to(device)
    observed_seismic = batch["observed_seismic"].to(device)
    predicted_velocity = model(cnn_input)
    if velocity_normalizer is not None:
        predicted_velocity = velocity_normalizer.denormalize(predicted_velocity)
    return {
        "predicted_velocity": predicted_velocity,
        "predicted_seismic": forward_operator(predicted_velocity),
        "observed_seismic": observed_seismic,
    }