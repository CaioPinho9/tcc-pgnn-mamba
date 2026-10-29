from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DEFAULT_DATASETS_ROOT, PHYSICS_CONFIG_DIR, load_physics_config
from physics.forward import build_forward_operator
from physics.geometry import build_acquisition_geometry


PHYSICS_CONFIG = PHYSICS_CONFIG_DIR / "marmousi.json"
DEFAULT_DATASET_DIR = DEFAULT_DATASETS_ROOT / "Marmousi"
SOURCE_FILENAMES = ("marmousi_bl.bin", "marmousi_ml.bin")


def _load_velocity(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")

    velocity = torch.as_tensor(value, dtype=torch.float32).squeeze()
    if velocity.ndim != 2:
        raise ValueError(f"Esperado um tensor de velocidade 2D em {path}; obtida a forma {tuple(velocity.shape)}")
    return velocity


def generate_marmousi_dataset(
        *,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        device: str = "cpu",
) -> tuple[Path, Path]:
    """Gera o modelo Marmousi original de 116x227 com a aquisição do PGNN-FWI."""
    physics_config = load_physics_config(PHYSICS_CONFIG)
    source_dir = dataset_dir / "source"
    velocities = torch.stack([_load_velocity(source_dir / filename) for filename in SOURCE_FILENAMES])

    model_shape = tuple(int(value) for value in velocities.shape[-2:])
    geometry = build_acquisition_geometry(physics_config, model_shape)
    if geometry is None:
        raise ValueError("O Marmousi exige uma geometria de aquisição explícita")

    receiver_indices = geometry["receiver_indices"]
    time_step_count = int(physics_config["recorded_time_total_steps"])
    forward_operator = build_forward_operator(
        source_indices=geometry["source_indices"],
        receiver_count=receiver_indices.shape[0],
        time_step_count=time_step_count,
        receiver_indices=receiver_indices,
        config=physics_config,
    ).to(device)

    observed_samples = []
    forward_operator.eval()
    with torch.no_grad():
        for index, velocity in enumerate(velocities):
            print(f"Simulando {SOURCE_FILENAMES[index]} em {device}...")
            observed = forward_operator(velocity[None, None].to(device))
            observed_samples.append(observed[0].cpu())

    seismic = torch.stack(observed_samples).numpy().astype(np.float32, copy=False)
    models = velocities[:, None].numpy().astype(np.float32, copy=False)

    data_path = dataset_dir / "data" / "data1.npy"
    model_path = dataset_dir / "model" / "model1.npy"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_path, seismic)
    np.save(model_path, models)

    print(f"Sísmica {seismic.shape} salva em {data_path}")
    print(f"Velocidade {models.shape} salva em {model_path}")
    return data_path, model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera observações Marmousi com o protocolo do PGNN-FWI."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    generate_marmousi_dataset(
        dataset_dir=DEFAULT_DATASET_DIR,
        device=args.device,
    )


if __name__ == "__main__":
    main()
