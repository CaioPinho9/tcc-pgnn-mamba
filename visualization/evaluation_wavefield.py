"""Comparação de campos de onda usada no notebook de avaliação dos modelos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display

from config.config import resolve_physics_config
from data.dataset import prepare_data
from network.evaluation import (
    EvaluationJob,
    EvaluationRun,
    _scenario_from_path,
    apply_evaluation_noise,
    build_run_config,
    find_checkpoint_paths,
    load_checkpoint_low_memory,
    predict_checkpoint,
)
from physics.forward import build_forward_operator
from physics.geometry import build_acquisition_geometry
from visualization.wavefield_animation import (
    WavefieldFrameRecorder,
    combine_wavefield_gifs,
    save_wavefield_gif,
)


def _select_unetmamba_scenario(factor_table: pd.DataFrame, preferred_geometry: str):
    scenarios = factor_table.loc[
        factor_table["model_name"].eq("unetmamba")
        & pd.to_numeric(factor_table["checkpoint_count"], errors="coerce").gt(0)
    ].copy()
    if scenarios.empty:
        return None

    clean_scenarios = scenarios.loc[
        pd.to_numeric(scenarios["noise_std_ratio"], errors="coerce").eq(0.0)
    ]
    if not clean_scenarios.empty:
        scenarios = clean_scenarios
    geometry_scenarios = scenarios.loc[
        scenarios["receiver_geometry"].eq(preferred_geometry)
    ]
    if not geometry_scenarios.empty:
        scenarios = geometry_scenarios
    noise_values = pd.to_numeric(scenarios["noise_std_ratio"], errors="coerce")
    scenarios = scenarios.loc[noise_values.eq(noise_values.min())]
    return scenarios.sort_values("epochs").iloc[-1]


def _scenario_label(scenario: pd.Series) -> str:
    noise_ratio = float(scenario["noise_std_ratio"])
    noise_label = "sem ruído" if noise_ratio == 0.0 else f"ruído {noise_ratio:g}"
    return (
        f"{scenario['model_label']} | {noise_label} | "
        f"{int(scenario['epochs'])} épocas | {scenario['geometry_description']}"
    )


def _find_scenario_checkpoint(
    scenario: pd.Series,
    *,
    dataset_name: str,
    sample_id: int,
    version: str,
    checkpoint_root: str | Path,
) -> Path | None:
    checkpoint_dir = Path(checkpoint_root)
    if checkpoint_dir.name != version:
        checkpoint_dir = checkpoint_dir / version
    for checkpoint_path in find_checkpoint_paths(
        "unetmamba",
        dataset_name,
        (checkpoint_dir,),
        noise_label=str(scenario["noise_label"]),
    ):
        factors = _scenario_from_path(checkpoint_path, "unetmamba", dataset_name)
        if (
            factors is not None
            and factors["scenario_id"] == scenario["scenario_id"]
            and factors["sample_id"] == sample_id
        ):
            return checkpoint_path
    return None


def _predict_scenario_sample(
    checkpoint_path: Path,
    scenario: pd.Series,
    *,
    dataset_name: str,
    sample_id: int,
    scenario_label: str,
    device: torch.device,
) -> np.ndarray:
    checkpoint = load_checkpoint_low_memory(checkpoint_path)
    run_config = build_run_config(dataset_name, "unetmamba")
    noise_ratio = float(scenario["noise_std_ratio"])
    noise_job = EvaluationJob(
        name="wavefield-animation",
        model_names=("unetmamba",),
        dataset_names=(dataset_name,),
        noise_std_ratio=noise_ratio,
    )
    run_config = apply_evaluation_noise(run_config, noise_job)
    run_config["seed"] = int(run_config.get("seed", 42))
    if checkpoint.get("physics_config") is not None:
        run_config["physics"] = checkpoint["physics_config"]
    torch.manual_seed(run_config["seed"])
    data = prepare_data(run_config, torch.device("cpu"), sample_index=sample_id)
    run = EvaluationRun(
        model_name="unetmamba",
        dataset_name=dataset_name,
        sample_index=sample_id,
        checkpoint_path=checkpoint_path,
        run_name=checkpoint_path.stem.removesuffix("_best"),
        model_label=scenario_label,
        noise_label=str(scenario["noise_label"]),
    )
    prediction, _, _ = predict_checkpoint(
        run,
        checkpoint,
        run_config,
        data,
        device,
        warmup_runs=0,
        timed_runs=1,
    )
    return np.asarray(prediction, dtype=np.float32)


def _capture_wavefield(operator, velocity: np.ndarray, device: torch.device, callback_frequency: int):
    recorder = WavefieldFrameRecorder()
    velocity_tensor = torch.as_tensor(
        velocity[None, None],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        operator(
            velocity_tensor,
            wavefield_callback=recorder,
            callback_frequency=callback_frequency,
        )
    return recorder


def display_unetmamba_wavefield_comparison(
    scenario_result,
    *,
    dataset_name: str,
    sample_id: int,
    version: str,
    checkpoint_root: str | Path,
    output_dir: str | Path,
    preferred_geometry: str,
    device: torch.device,
    callback_frequency: int = 10,
) -> Path | None:
    """Gera e exibe GIFs comparativos do campo de onda."""
    scenario = _select_unetmamba_scenario(
        scenario_result.factor_table,
        preferred_geometry,
    )
    if scenario is None:
        display(Markdown("Não há checkpoints U-Net Mamba disponíveis para gerar a animação."))
        return None

    scenario_label = _scenario_label(scenario)
    predicted_velocity = scenario_result.output.predictions_by_sample.get(
        dataset_name,
        {},
    ).get(sample_id, {}).get(scenario_label)
    if predicted_velocity is None:
        checkpoint_path = _find_scenario_checkpoint(
            scenario,
            dataset_name=dataset_name,
            sample_id=sample_id,
            version=version,
            checkpoint_root=checkpoint_root,
        )
        if checkpoint_path is None:
            display(Markdown(
                f"Não encontrei o checkpoint U-Net Mamba do cenário `{scenario_label}` "
                f"para a amostra {sample_id}."
            ))
            return None
        predicted_velocity = _predict_scenario_sample(
            checkpoint_path,
            scenario,
            dataset_name=dataset_name,
            sample_id=sample_id,
            scenario_label=scenario_label,
            device=device,
        )

    reference_config = build_run_config(dataset_name, "pgnn_mardan_2024")
    torch.manual_seed(int(reference_config.get("seed", 42)))
    reference_data = prepare_data(
        reference_config,
        torch.device("cpu"),
        sample_index=sample_id,
    )
    physics = reference_data["physics_config"] or resolve_physics_config(reference_config)
    geometry = build_acquisition_geometry(
        physics,
        tuple(int(value) for value in reference_data["raw_y"].shape[-2:]),
    )
    operator = build_forward_operator(
        source_indices=geometry["source_indices"],
        receiver_count=int(geometry["receiver_indices"].shape[0]),
        time_step_count=int(physics["recorded_time_total_steps"]),
        receiver_indices=geometry["receiver_indices"],
        receiver_z_index=1,
        config=physics,
    ).to(device)
    true_velocity = np.asarray(reference_data["raw_y"][0, 0]).copy()
    predicted_velocity = np.asarray(predicted_velocity, dtype=np.float32)

    reference_path = (
        Path("results/data_analysis/figures/seismic_acquisition")
        / f"{dataset_name}_s{sample_id:04d}_forward_wavefield.gif"
    )
    output_dir = Path(output_dir) / "figures" / "seismic_acquisition"
    predicted_path = output_dir / f"{dataset_name}_s{sample_id:04d}_unetmamba_wavefield.gif"
    comparison_path = output_dir / f"{dataset_name}_s{sample_id:04d}_reference_vs_unetmamba.gif"

    save_wavefield_gif(
        _capture_wavefield(operator, true_velocity, device, callback_frequency),
        reference_path,
        grid_spacing_m=float(physics["grid_spacing_m"]),
    )
    save_wavefield_gif(
        _capture_wavefield(operator, predicted_velocity, device, callback_frequency),
        predicted_path,
        grid_spacing_m=float(physics["grid_spacing_m"]),
    )
    epochs = int(scenario["epochs"])
    combine_wavefield_gifs(
        reference_path,
        predicted_path,
        comparison_path,
        left_label="Propagação usando velocidade de referência",
        right_label=f"Propagação usando predição U-Net Mamba — {epochs} épocas",
    )
    print("Cenário U-Net Mamba:", scenario_label)
    print("GIF da predição:", predicted_path)
    print("Comparação lado a lado:", comparison_path)
    display(Image(filename=str(comparison_path)))
    return comparison_path
