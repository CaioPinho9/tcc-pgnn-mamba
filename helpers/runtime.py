from __future__ import annotations

import os
import time
from statistics import mean, pstdev
from typing import Iterable

import torch


def prepare_torch_extension_cache(cache_dir: str) -> None:
    """Configura um cache privado para extensões e remove um bloqueio antigo de compilação."""
    if "TORCH_EXTENSIONS_DIR" in os.environ:
        return

    os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
    lock_path = os.path.join(cache_dir, "deepwave", "lock")
    try:
        if time.time() - os.path.getmtime(lock_path) > 300:
            os.unlink(lock_path)
    except OSError:
        pass


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_gpu_memory_mb(device: torch.device) -> float:
    """Retorna o pico de memória alocada em megabytes decimais."""
    if device.type != "cuda":
        return 0.0
    allocated = torch.cuda.max_memory_allocated(device)
    return float(allocated / 1_000_000)


def gpu_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return max(1, torch.cuda.device_count())


def measure_inference_time(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    enabled: bool = True,
    max_batches: int = 8,
) -> dict[str, float | int]:
    if not enabled:
        return {
            "mean_inference_time_ms": 0.0,
            "std_inference_time_ms": 0.0,
            "num_inference_samples": 0,
        }

    model.eval()
    times_ms: list[float] = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            if batch_idx >= max_batches:
                break

            cnn_input = batch["cnn_input"].to(device)
            synchronize_if_cuda(device)
            start = time.perf_counter()
            _ = model(cnn_input)
            synchronize_if_cuda(device)
            elapsed = time.perf_counter() - start

            times_ms.append(elapsed * 1000.0)
            sample_count += int(cnn_input.shape[0])

    return {
        "mean_inference_time_ms": mean(times_ms) if times_ms else 0.0,
        "std_inference_time_ms": pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "num_inference_samples": sample_count,
    }
