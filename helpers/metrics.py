from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def compute_mae(pred, true) -> float:
    pred_np = _as_numpy(pred)
    true_np = _as_numpy(true)
    return float(np.mean(np.abs(pred_np - true_np)))


def compute_ssim(pred, true, data_range: float | None = None) -> float:
    pred_np = _as_numpy(pred)
    true_np = _as_numpy(true)
    if data_range is None:
        data_range = float(np.max(true_np) - np.min(true_np))
    data_range = float(data_range)
    if not np.isfinite(data_range) or data_range < 0:
        raise ValueError("SSIM data_range must be finite and non-negative.")
    if data_range == 0:
        return 1.0 if np.array_equal(pred_np, true_np) else 0.0

    return float(structural_similarity(
        pred_np,
        true_np,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    ))
