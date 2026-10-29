from __future__ import annotations
from scipy.signal import butter, freqz

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=64)
def _butterworth_response(
        time_steps: int,
        cutoff_hz: float,
        sample_rate_hz: float,
        order: int,
):
    torch.set_flush_denormal(False)

    normalized_cutoff = 2.0 * cutoff_hz / sample_rate_hz
    numerator, denominator = butter(
        order,
        normalized_cutoff,
        btype="lowpass",
        output="ba",
    )
    _, response = freqz(numerator, denominator, worN=time_steps, whole=True)
    return response.astype("complex64", copy=False)


def lowpass_seismic(
        seismic: torch.Tensor,
        cutoff_hz: float | None,
        time_spacing_s: float,
        *,
        sample_rate_hz: float | None = None,
        order: int = 3,
        padding: int = 512,
) -> torch.Tensor:
    if cutoff_hz is None:
        return seismic

    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz deve ser positivo; obtido: {cutoff_hz}")

    time_steps = seismic.shape[2]
    if seismic.numel() == 0:
        return seismic

    sample_rate_hz = (
        1.0 / float(time_spacing_s)
        if sample_rate_hz is None
        else float(sample_rate_hz)
    )
    order = int(order)
    padding = int(padding)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz deve ser positivo")
    if not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("cutoff_hz deve estar abaixo da frequência de Nyquist")
    if order <= 0:
        raise ValueError("order deve ser positivo")
    if padding < 0:
        raise ValueError("padding não pode ser negativo")

    padded = F.pad(seismic, (0, 0, 0, padding))
    response_values = _butterworth_response(
        padded.shape[2], cutoff_hz, sample_rate_hz, order
    )
    response_dtype = torch.complex128 if seismic.dtype == torch.float64 else torch.complex64
    response = torch.as_tensor(
        response_values,
        dtype=response_dtype,
        device=seismic.device,
    ).view(1, 1, -1, 1)
    spectrum = torch.fft.fft(padded, dim=2)
    # Reproduza o par filtro direto/filtro adjunto usado por pinn_fwi.
    filtered = torch.fft.ifft(spectrum * response * response.conj(), dim=2).real
    return filtered[:, :, :time_steps, :]
