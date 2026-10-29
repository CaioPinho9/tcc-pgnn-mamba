from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.config import resolve_physics_config
from physics.geometry import build_acquisition_geometry
from physics.forward import build_forward_operator
from .files import load_openfwi_arrays, openfwi_sample_count
from .seismic_noise import add_gaussian_noise
from transformation.min_max_normalizer import MinMaxNormalizer


class OpenFWIDataset(Dataset):
    """Para inversão CNN não supervisionada guiada pela física.

        raw_seismic:
            [sample, source, time, receiver]
        normalizer:
            Objeto Normalizer para dados sísmicos ou None para não aplicar normalização.

        Retorna:
            cnn_input:
                Dados sísmicos (opcionalmente normalizados), [source, time, receiver]

            observed_seismic:
                Dados sísmicos usados na perda física, [source, time, receiver]

            target_velocity:
                Alvo de velocidade bruto, [channel, z, x], quando fornecido
    """

    def __init__(
            self,
            raw_seismic,
            target_velocity=None,
            normalizer=None,
            input_seismic=None,
            observed_seismic=None,
    ):
        self.raw_seismic = torch.as_tensor(raw_seismic, dtype=torch.float32)
        self.input_seismic = (
            self.raw_seismic
            if input_seismic is None
            else torch.as_tensor(input_seismic, dtype=torch.float32)
        )
        self.observed_seismic = (
            self.raw_seismic
            if observed_seismic is None
            else torch.as_tensor(observed_seismic, dtype=torch.float32)
        )
        self.target_velocity = (
            torch.as_tensor(target_velocity, dtype=torch.float32)
            if target_velocity is not None
            else None
        )

        if self.raw_seismic.ndim != 4:
            raise ValueError(
                "raw_seismic deve ter a forma "
                "[sample, source, time, receiver]"
            )
        if self.input_seismic.shape != self.raw_seismic.shape:
            raise ValueError("input_seismic deve ter a mesma forma que raw_seismic")
        if self.observed_seismic.shape != self.raw_seismic.shape:
            raise ValueError("observed_seismic deve ter a mesma forma que raw_seismic")
        if self.target_velocity is not None and self.target_velocity.shape[0] != self.raw_seismic.shape[0]:
            raise ValueError("target_velocity e raw_seismic devem ter a mesma quantidade de amostras")
        self.normalizer = normalizer

    def __len__(self):
        return self.raw_seismic.shape[0]

    def __getitem__(self, index):
        observed_seismic = self.observed_seismic[index]
        cnn_input = self.input_seismic[index]
        if self.normalizer:
            cnn_input = self.normalizer.normalize(self.input_seismic[index])

        sample = {
            "cnn_input": cnn_input,
            "observed_seismic": observed_seismic,
            "sample_index": index,
        }
        if self.target_velocity is not None:
            sample["target_velocity"] = self.target_velocity[index]
        return sample


def _synthesize_receiver_data_from_velocity(
        target_velocity: np.ndarray,
        physics_config: dict[str, Any],
        acquisition_geometry: dict[str, np.ndarray],
        time_step_count: int,
        device: torch.device,
) -> np.ndarray:
    if target_velocity.shape[0] != 1:
        raise ValueError(
            "A síntese de receptores adicionais exige preparar uma amostra por vez; "
            "use sample_index para OpenFWI com surface_wells."
        )

    receiver_indices = acquisition_geometry["receiver_indices"]
    forward_operator = build_forward_operator(
        source_indices=acquisition_geometry["source_indices"],
        receiver_count=int(receiver_indices.shape[0]),
        time_step_count=time_step_count,
        receiver_indices=receiver_indices,
        checkpoint_time_steps=physics_config.get("checkpoint_time_steps"),
        config=physics_config,
    ).to(device)
    velocity_tensor = torch.as_tensor(target_velocity, dtype=torch.float32, device=device)
    if velocity_tensor.ndim == 3:
        velocity_tensor = velocity_tensor.unsqueeze(1)
    with torch.no_grad():
        seismic = forward_operator(velocity_tensor)
    return seismic.detach().cpu().numpy()


def prepare_data(
        config: dict[str, Any],
        device: torch.device,
        *,
        sample_index: int | None = None,
) -> dict[str, Any]:
    physics_config = resolve_physics_config(config)
    total_samples = None
    if sample_index is not None:
        total_samples = openfwi_sample_count(config["datasets_root"], config["dataset_name"])
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= total_samples:
            raise IndexError(f"sample_index {sample_index} está fora do intervalo do dataset 0..{total_samples - 1}")

    # raw_x guarda os sismogramas observados; raw_y é o modelo de velocidade
    # correspondente.
    raw_x, raw_y = load_openfwi_arrays(
        config["datasets_root"],
        config["dataset_name"],
        config.get("max_samples") if sample_index is None else None,
        sample_indices=[sample_index] if sample_index is not None else None,
    )
    model_shape = tuple(int(value) for value in raw_y.shape[-2:])
    acquisition_geometry = build_acquisition_geometry(physics_config, model_shape)
    receiver_selection = None

    if acquisition_geometry is not None:
        expected_receivers = int(acquisition_geometry["receiver_indices"].shape[0])
        receiver_data_geometry = physics_config.get("receiver_data_geometry")
        if raw_x.shape[-1] != expected_receivers and receiver_data_geometry:
            source_physics = dict(physics_config)
            source_physics["receiver_geometry"] = receiver_data_geometry
            source_geometry = build_acquisition_geometry(source_physics, model_shape)
            source_receiver_indices = source_geometry["receiver_indices"]
            if raw_x.shape[-1] == source_receiver_indices.shape[0]:
                source_lookup = {
                    (int(x), int(z)): index
                    for index, (x, z) in enumerate(source_receiver_indices)
                }
                try:
                    receiver_selection = [
                        source_lookup[(int(x), int(z))]
                        for x, z in acquisition_geometry["receiver_indices"]
                    ]
                except KeyError:
                    receiver_selection = None

        if raw_x.shape[-1] != expected_receivers and receiver_selection is None:
            if physics_config.get("synthesize_receiver_data_from_velocity"):
                raw_x = _synthesize_receiver_data_from_velocity(
                    raw_y,
                    physics_config,
                    acquisition_geometry,
                    int(raw_x.shape[-2]),
                    device,
                )

    expected_sources = int(physics_config["source_count"])
    if raw_x.shape[1] != expected_sources:
        raise ValueError(
            f"O dataset possui {raw_x.shape[1]} canais de fonte, mas sua configuração física "
            f"declara source_count={expected_sources}"
        )
    expected_time_steps = int(physics_config.get("recorded_time_total_steps", raw_x.shape[-2]))
    if raw_x.shape[-2] != expected_time_steps:
        raise ValueError(
            f"O dataset possui {raw_x.shape[-2]} passos temporais, mas sua configuração física "
            f"declara recorded_time_total_steps={expected_time_steps}"
        )
    if acquisition_geometry is not None:
        expected_receivers = int(acquisition_geometry["receiver_indices"].shape[0])
        if raw_x.shape[-1] != expected_receivers and receiver_selection is None:
            raise ValueError(
                f"O dataset possui {raw_x.shape[-1]} receptores, mas sua geometria de aquisição "
                f"contém {expected_receivers} posições de receptores"
            )

    if sample_index is None:
        all_indices = np.arange(raw_x.shape[0])
    else:
        all_indices = np.arange(total_samples)

    if sample_index is not None:
        sample_indices = np.array([sample_index], dtype=int)
    else:
        sample_indices = all_indices

    raw_seismic = torch.as_tensor(raw_x, dtype=torch.float32)
    training = config.get("training", {})
    input_noise_std_ratio = training.get("seismic_input_noise_std_ratio")
    observed_noise_std_ratio = training.get("seismic_observed_noise_std_ratio")

    # Separa o ruído de entrada da rede do ruído dos dados usados na perda física.
    if (
            input_noise_std_ratio is not None
            and observed_noise_std_ratio is not None
            and float(input_noise_std_ratio) == float(observed_noise_std_ratio)
    ):
        noisy_seismic = add_gaussian_noise(raw_seismic, input_noise_std_ratio)
        input_seismic = noisy_seismic
        observed_seismic = noisy_seismic
    else:
        input_seismic = add_gaussian_noise(raw_seismic, input_noise_std_ratio)
        observed_seismic = add_gaussian_noise(raw_seismic, observed_noise_std_ratio)

    if receiver_selection is not None:
        raw_x = raw_x[..., receiver_selection]
        raw_seismic = raw_seismic[..., receiver_selection]
        input_seismic = input_seismic[..., receiver_selection]
        observed_seismic = observed_seismic[..., receiver_selection]

    normalizer = MinMaxNormalizer().fit(observed_seismic)

    # Cada amostra fornece os traços de entrada à rede e os observados à perda;
    # o modelo de velocidade de referência fica disponível para análise estrutural.
    dataset = OpenFWIDataset(
        raw_seismic=raw_seismic,
        input_seismic=input_seismic,
        observed_seismic=observed_seismic,
        target_velocity=raw_y,
        normalizer=normalizer,
    )
    dataloader_config = config.get("dataloader") or {}
    batch_size = (
        1
        if sample_index is not None
        else int(dataloader_config.get("batch_size", 1))
    )
    num_workers = int(dataloader_config.get("num_workers", 0))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "raw_x": raw_x,
        "input_x": input_seismic.numpy(),
        "observed_x": observed_seismic.numpy(),
        "raw_y": raw_y,
        "all_indices": all_indices,
        "sample_indices": sample_indices,
        "normalizer": normalizer,
        "loader": loader,
        "model_shape": model_shape,
        "receiver_count": raw_x.shape[-1],
        "time_step_count": raw_x.shape[-2],
        "physics_config": physics_config,
        "source_indices": (
            acquisition_geometry["source_indices"] if acquisition_geometry is not None else None
        ),
        "receiver_indices": (
            acquisition_geometry["receiver_indices"] if acquisition_geometry is not None else None
        ),
    }


def configured_sample_indices(config: dict[str, Any]) -> np.ndarray:
    total_samples = openfwi_sample_count(config["datasets_root"], config["dataset_name"])
    configured_indices = config.get("sample_indices")
    if configured_indices is not None:
        sample_indices = np.atleast_1d(configured_indices).astype(int)
        if np.any(sample_indices < 0) or np.any(sample_indices >= total_samples):
            raise IndexError(f"sample_indices devem estar dentro do intervalo do dataset 0..{total_samples - 1}")
        return sample_indices

    max_samples = config.get("max_samples")
    if max_samples is None:
        sample_count = total_samples
    else:
        sample_count = min(int(max_samples), total_samples)

    if sample_count <= 0:
        return np.array([], dtype=int)

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    return rng.permutation(total_samples)[:sample_count].astype(int)


def prepare_training_sample(data: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    if len(data["sample_indices"]) != 1:
        raise ValueError(
            "O treinamento espera uma execução com uma única amostra. "
            "Use prepare_data(config, device, sample_index=...) para cada modelo independente."
        )

    try:
        batch = next(iter(data["loader"]))
    except StopIteration as exc:
        raise ValueError("O carregador de dados de treinamento está vazio.") from exc

    # Reúna a amostra selecionada e mova os tensores para o dispositivo do treino.
    return {
        "cnn_input": batch["cnn_input"].to(device),
        "observed_seismic": batch["observed_seismic"].to(device),
        "target_velocity": batch["target_velocity"].to(device),
    }