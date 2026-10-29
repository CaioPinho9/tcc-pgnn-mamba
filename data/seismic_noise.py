from __future__ import annotations

import math

import torch


def add_gaussian_noise(seismic: torch.Tensor, std_ratio: float | None) -> torch.Tensor:
    """Adiciona ruído gaussiano branco proporcional ao desvio padrão sísmico."""
    if std_ratio is None:
        return seismic

    std_ratio = float(std_ratio)
    if not math.isfinite(std_ratio) or std_ratio < 0.0:
        raise ValueError("std_ratio must be a finite non-negative number")

    signal_std = seismic.std(unbiased=False)
    if signal_std.item() == 0.0 or std_ratio == 0.0:
        return seismic.clone()

    return seismic + torch.randn_like(seismic) * (signal_std * std_ratio)


def format_noise_ratio(std_ratio: float | None) -> str:
    """Formata uma razão de ruído para manter nomes de execução e pastas estáveis."""
    if std_ratio is None:
        return "clean"
    value = float(std_ratio)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("std_ratio must be a finite non-negative number")
    text = format(value, ".12g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def noise_condition_label(
        input_std_ratio: float | None,
        observed_std_ratio: float | None,
) -> str:
    """Retorna um rótulo seguro para nome de arquivo que identifica o ruído na entrada e na perda."""
    if input_std_ratio is None and observed_std_ratio is None:
        return "clean"
    if (
            input_std_ratio is not None
            and observed_std_ratio is not None
            and float(input_std_ratio) == float(observed_std_ratio)
    ):
        return f"noise_{format_noise_ratio(input_std_ratio)}"
    return (
        f"input_{format_noise_ratio(input_std_ratio)}"
        f"_observed_{format_noise_ratio(observed_std_ratio)}"
    )


def noise_run_suffix(
        input_std_ratio: float | None,
        observed_std_ratio: float | None,
) -> str:
    """Retorna o sufixo opcional do nome da execução para uma condição de ruído."""
    label = noise_condition_label(input_std_ratio, observed_std_ratio)
    return "" if label == "clean" else f"_{label}"
