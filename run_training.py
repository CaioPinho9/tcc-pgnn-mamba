"""Executa treinamentos em sequência, sem abrir um notebook.

Exemplos
--------
Sem argumentos, selecione modelos e conjuntos de dados nos menus do terminal::

    python run_training.py

Treine dois modelos nos conjuntos de dados OpenFWI disponíveis::

    python run_training.py --models mamba unet --datasets FlatFault_A CurveVel_A CurveFault_B

Use ruído com desvio padrão igual a 0,5 vez o desvio padrão sísmico::

    python run_training.py --models mamba --datasets Marmousi --noise 0.5

Execute as combinações de ruído, épocas e geometria selecionadas::

    python run_training.py --models mamba --datasets Marmousi --noise 0 0.5 --epochs 500 2500 --geometry surface surface_wells --product

O argumento ``--product`` combina os valores selecionados para cada conjunto de dados; ele não adiciona
outros conjuntos de dados à seleção. Todos aceitam as geometrias ``surface`` e
``surface_wells``. No OpenFWI, o simulador local sintetiza os traços da geometria selecionada
a partir do modelo de velocidade, pois os arquivos originais contêm apenas receptores de superfície.

Por padrão, cada conjunto de dados OpenFWI usa as amostras de índices 0 e 1. Marmousi mantém a seleção
definida em sua configuração.

Quando ``--datasets`` é omitido, cada modelo usa os conjuntos de dados definidos em sua configuração
e os conjuntos compatíveis encontrados na raiz de dados. A opção ``--sample-indices`` substitui
a seleção padrão das duas primeiras amostras do OpenFWI. Checkpoints concluídos são ignorados;
execuções parciais são retomadas.
"""

from __future__ import annotations

import argparse
from collections import deque
from itertools import product
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from config.config import (
    DEFAULT_CHECKPOINT_ROOT,
    MODEL_CONFIGS,
    PROJECT_ROOT,
    config_for_dataset,
    load_training_config,
    resolve_device,
)
from data.dataset import configured_sample_indices
from data.files import openfwi_sample_count, resolve_dataset_files
from data.paths import ensure_output_dirs
from data.seismic_noise import format_noise_ratio, noise_condition_label
from network.training_runner import (
    pending_training_sample_indices,
    release_memory,
    run_training_samples,
)


OPENFWI_DATASET_NAMES = (
    "FlatFault_A",
    "CurveVel_A",
    "CurveFault_B",
)
SUPPORTED_DATASET_NAMES = (*OPENFWI_DATASET_NAMES, "Marmousi")
OPENFWI_DEFAULT_SAMPLE_COUNT = 2
MAX_RETRIES_PER_COMBINATION = 1
DATASET_RECEIVER_GEOMETRIES = {
    **{name: ("surface", "surface_wells") for name in OPENFWI_DATASET_NAMES},
    "Marmousi": ("surface", "surface_wells"),
}


@dataclass(frozen=True)
class TrainingJob:
    model_name: str
    config_path: Path
    dataset_names: tuple[str, ...] | None


@dataclass(frozen=True)
class RunScenario:
    noise_std_ratio: float | None
    epochs: int | None
    receiver_geometry: str | None
    noise_selected: bool
    epochs_selected: bool
    geometry_selected: bool

    @property
    def output_label(self) -> str:
        parts = []
        if self.noise_selected:
            noise_label = "clean" if self.noise_std_ratio == 0.0 else format_noise_ratio(self.noise_std_ratio)
            parts.append(f"noise_{noise_label}")
        if self.epochs_selected:
            parts.append(f"epochs_{self.epochs}")
        if self.geometry_selected:
            parts.append(f"geometry_{self.receiver_geometry}")
        return "__".join(parts)


@dataclass(frozen=True)
class TrainingCombination:
    job: TrainingJob
    dataset_name: str
    scenario: RunScenario
    sample_indices: tuple[int, ...] | None = None


class TotalETA:
    """Estima o tempo restante da fila com base nas épocas de treinamento concluídas."""

    def __init__(self, total_epochs: int):
        self.total_epochs = max(0, int(total_epochs))
        self.remaining_epochs = self.total_epochs
        self.completed_epochs = 0
        self.timed_epochs = 0
        self.epoch_seconds = 0.0
        self.active_task_epochs: int | None = None
        self.active_task_started = False
        self.active_task_callbacks = 0
        self.active_callback_time = 0.0

    def start_task(self, epochs: int) -> None:
        self.active_task_epochs = max(0, int(epochs))
        self.active_task_started = False
        self.active_task_callbacks = 0
        self.active_callback_time = time.perf_counter()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def on_epoch(self, row: dict[str, Any]) -> None:
        if self.active_task_epochs is None:
            return

        epoch = int(row.get("epoch", 0))
        if not self.active_task_started:
            resumed_epochs = min(self.active_task_epochs, max(0, epoch - 1))
            self.remaining_epochs = max(0, self.remaining_epochs - resumed_epochs)
            self.completed_epochs += resumed_epochs
            self.active_task_started = True

        self.remaining_epochs = max(0, self.remaining_epochs - 1)
        self.completed_epochs += 1
        now = time.perf_counter()
        if self.active_task_callbacks:
            elapsed = now - self.active_callback_time
        else:
            elapsed = float(row.get("total_epoch_time_seconds", row.get("epoch_seconds", 0.0)) or 0.0)
        self.active_task_callbacks += 1
        self.active_callback_time = now
        if elapsed > 0.0:
            self.epoch_seconds += elapsed
            self.timed_epochs += 1

        average_epoch_seconds = self.epoch_seconds / self.timed_epochs if self.timed_epochs else 0.0
        row["total_eta"] = self._format_duration(average_epoch_seconds * self.remaining_epochs)
        row["total_epoch_progress"] = f"{self.completed_epochs}/{self.total_epochs}"

    def finish_task(self, *, skipped: bool = False) -> None:
        if self.active_task_epochs is None:
            return
        if skipped:
            self.remaining_epochs = max(0, self.remaining_epochs - self.active_task_epochs)
            self.completed_epochs += self.active_task_epochs
        self.active_task_epochs = None
        self.active_task_started = False
        self.active_task_callbacks = 0

    @property
    def eta(self) -> str:
        if self.remaining_epochs == 0:
            return self._format_duration(0.0)
        if not self.timed_epochs:
            return "estimating"
        average_epoch_seconds = self.epoch_seconds / self.timed_epochs
        return self._format_duration(average_epoch_seconds * self.remaining_epochs)


def build_scenarios(args: argparse.Namespace) -> list[RunScenario]:
    noise_values = args.noise_ratios if args.noise_ratios is not None else [None]
    epoch_values = args.epoch_values if args.epoch_values is not None else [None]
    geometry_values = args.receiver_geometries if args.receiver_geometries is not None else [None]
    selected_lengths = (len(noise_values), len(epoch_values), len(geometry_values))
    if not args.product and any(length > 1 for length in selected_lengths):
        raise ValueError("Utilize --product para combinar todos os valores selecionados entre si.")

    return [
        RunScenario(
            noise_std_ratio=noise,
            epochs=epochs,
            receiver_geometry=geometry,
            noise_selected=args.noise_ratios is not None,
            epochs_selected=args.epoch_values is not None,
            geometry_selected=args.receiver_geometries is not None,
        )
        for noise, epochs, geometry in product(noise_values, epoch_values, geometry_values)
    ]


def _scenario_results_version(base_version: str, scenario: RunScenario) -> str:
    return f"{base_version}/{scenario.output_label}" if scenario.output_label else base_version


def _describe_scenario(scenario: RunScenario) -> str:
    noise = (
        "configurado"
        if not scenario.noise_selected
        else "sem ruído" if scenario.noise_std_ratio == 0.0 else f"{scenario.noise_std_ratio:g}"
    )
    epochs = str(scenario.epochs) if scenario.epochs_selected else "configuradas"
    geometry = scenario.receiver_geometry if scenario.geometry_selected else "configurada"
    return f"ruído={noise}, épocas={epochs}, geometria={geometry}"


def _scenarios_for_dataset(
        scenarios: Sequence[RunScenario],
        dataset_name: str,
) -> list[RunScenario]:
    supported_geometries = DATASET_RECEIVER_GEOMETRIES[dataset_name]
    return [
        scenario
        for scenario in scenarios
        if not scenario.geometry_selected
        or scenario.receiver_geometry in supported_geometries
    ]


def _validate_scenario_compatibility(
        jobs: Sequence[TrainingJob],
        scenarios: Sequence[RunScenario],
) -> None:
    has_compatible_scenario = False
    for job in jobs:
        config = load_training_config(job.config_path)
        dataset_names = job.dataset_names or _available_dataset_names(
            config,
        )
        _validate_supported_dataset_names(dataset_names)
        for dataset_name in dataset_names:
            compatible_scenarios = _scenarios_for_dataset(scenarios, dataset_name)
            has_compatible_scenario = has_compatible_scenario or bool(compatible_scenarios)
            skipped_geometries = tuple(dict.fromkeys(
                str(scenario.receiver_geometry)
                for scenario in scenarios
                if scenario.geometry_selected
                and scenario.receiver_geometry not in DATASET_RECEIVER_GEOMETRIES[dataset_name]
            ))
            if skipped_geometries:
                print(
                    f"Ignorando a geometria {', '.join(skipped_geometries)} para {dataset_name}; "
                    f"opções compatíveis: {', '.join(DATASET_RECEIVER_GEOMETRIES[dataset_name])}."
                )
    if not has_compatible_scenario:
        raise ValueError("Nenhuma combinação de conjunto de dados e geometria é compatível com a seleção informada.")


def _configure_combination(
        combination: TrainingCombination,
        args: argparse.Namespace,
) -> tuple[dict[str, Any], list[int]]:
    """Resolve a configuração, o nome da execução, a pasta de checkpoints e as amostras da tarefa."""
    job = combination.job
    dataset_name = combination.dataset_name
    scenario = combination.scenario
    config = load_training_config(job.config_path)
    _validate_datasets(config, (dataset_name,))
    run_config = config_for_dataset(config, dataset_name, model_name=job.model_name)
    run_config["results_version"] = _scenario_results_version(args.results_version, scenario)
    checkpointing = run_config.setdefault("checkpointing", {})
    configured_checkpoint_root = Path(
        checkpointing.get("dir", str(DEFAULT_CHECKPOINT_ROOT))
    )
    checkpoint_root = (
        DEFAULT_CHECKPOINT_ROOT
        if not configured_checkpoint_root.is_absolute()
        else configured_checkpoint_root
    )
    checkpointing["dir"] = str(checkpoint_root / args.results_version)
    if scenario.noise_selected:
        noise_ratio = None if scenario.noise_std_ratio == 0.0 else scenario.noise_std_ratio
        run_config.setdefault("training", {})["seismic_input_noise_std_ratio"] = noise_ratio
        run_config.setdefault("training", {})["seismic_observed_noise_std_ratio"] = noise_ratio
    if scenario.epochs_selected:
        run_config.setdefault("training", {})["epochs"] = scenario.epochs
    if scenario.geometry_selected:
        physics = run_config.setdefault("physics", {})
        physics["receiver_geometry"] = (
            "surface" if scenario.receiver_geometry == "surface" else "surface_and_wells"
        )

    run_label_parts = []
    has_scenario = scenario.noise_selected or scenario.epochs_selected or scenario.geometry_selected
    if has_scenario and noise_condition_label(
        run_config.get("training", {}).get("seismic_input_noise_std_ratio"),
        run_config.get("training", {}).get("seismic_observed_noise_std_ratio"),
    ) == "clean":
        run_label_parts.append("clean")
    if scenario.epochs_selected:
        run_label_parts.append(f"epochs{scenario.epochs}")
    if scenario.geometry_selected:
        run_label_parts.append(str(scenario.receiver_geometry))
    run_config["run_label"] = "_".join(run_label_parts)
    _configure_sample_selection(run_config, dataset_name, args)
    return run_config, configured_sample_indices(run_config)


def _remove_completed_checkpoints(
        combinations: Sequence[TrainingCombination],
        args: argparse.Namespace,
) -> list[TrainingCombination]:
    """Mantém na fila apenas as amostras sem checkpoint concluído."""
    pending_combinations = []
    for combination in combinations:
        run_config, sample_indices = _configure_combination(combination, args)
        pending_indices = pending_training_sample_indices(
            combination.job.model_name,
            run_config,
            sample_indices,
            force=args.force,
        )
        completed_count = len(sample_indices) - len(pending_indices)
        if completed_count:
            print(
                f"Removendo {completed_count} checkpoint(s) concluído(s) da fila para "
                f"modelo={combination.job.model_name}, conjunto de dados={combination.dataset_name}."
            )
        if pending_indices:
            pending_combinations.append(TrainingCombination(
                job=combination.job,
                dataset_name=combination.dataset_name,
                scenario=combination.scenario,
                sample_indices=tuple(pending_indices),
            ))
    return pending_combinations


def _planned_epoch_count(
        combinations: Sequence[TrainingCombination],
        args: argparse.Namespace,
) -> tuple[int, int]:
    total_epochs = 0
    training_tasks = 0
    for combination in combinations:
        config = load_training_config(combination.job.config_path)
        run_config = config_for_dataset(
            config,
            combination.dataset_name,
            model_name=combination.job.model_name,
        )
        if combination.scenario.epochs_selected:
            run_config.setdefault("training", {})["epochs"] = combination.scenario.epochs
        _configure_sample_selection(run_config, combination.dataset_name, args)
        sample_count = (
            len(combination.sample_indices)
            if combination.sample_indices is not None
            else len(configured_sample_indices(run_config))
        )
        epochs = int(run_config["training"]["epochs"])
        total_epochs += epochs * sample_count
        training_tasks += sample_count

    return total_epochs, training_tasks


def _expand_dataset_combinations(
        jobs: Sequence[TrainingJob],
        scenarios: Sequence[RunScenario],
) -> list[TrainingCombination]:
    """Expande a fila por conjunto de dados, preservando a ordem das tarefas e dos cenários."""
    jobs_by_dataset: dict[str, list[TrainingJob]] = {}
    for job in jobs:
        config = load_training_config(job.config_path)
        dataset_names = job.dataset_names or _available_dataset_names(config)
        for dataset_name in dataset_names:
            jobs_by_dataset.setdefault(dataset_name, []).append(job)

    combinations: list[TrainingCombination] = []
    for dataset_name, dataset_jobs in jobs_by_dataset.items():
        for job in dataset_jobs:
            for scenario in _scenarios_for_dataset(scenarios, dataset_name):
                combinations.append(TrainingCombination(
                    job=job,
                    dataset_name=dataset_name,
                    scenario=scenario,
                ))
    return combinations


def _split_values(values: Sequence[str] | None) -> list[str]:
    """Combina valores repetidos ou separados por vírgulas recebidos na linha de comando."""
    result: list[str] = []
    for value in values or []:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _resolve_config_path(model_name: str) -> Path:
    return MODEL_CONFIGS[model_name]


def _config_dataset_names(config: dict[str, Any]) -> tuple[str, ...]:
    names = config.get("dataset_names") or ("Marmousi",)
    return tuple(_canonical_dataset_name(str(name)) for name in names)


def _canonical_dataset_name(dataset_name: str) -> str:
    supported_names = {name.casefold(): name for name in SUPPORTED_DATASET_NAMES}
    return supported_names.get(dataset_name.strip().casefold(), dataset_name.strip())


def _available_dataset_names(
        config: dict[str, Any],
        *,
        include_missing_openfwi: bool = False,
) -> tuple[str, ...]:
    """Retorna os conjuntos de dados configurados e compatíveis encontrados na raiz de dados."""
    names = [
        name
        for name in _config_dataset_names(config)
        if name in SUPPORTED_DATASET_NAMES
    ]
    for dataset_name in SUPPORTED_DATASET_NAMES:
        if dataset_name in names:
            continue
        if dataset_name in OPENFWI_DATASET_NAMES and include_missing_openfwi:
            names.append(dataset_name)
            continue
        try:
            _validate_datasets(config, (dataset_name,))
        except (FileNotFoundError, ValueError, KeyError):
            continue
        names.append(dataset_name)
    return tuple(names)


def build_jobs(args: argparse.Namespace) -> list[TrainingJob]:
    selected_models = [name.strip().lower() for name in _split_values(args.models)]
    if not selected_models:
        raise ValueError("Selecione pelo menos um modelo usando --models.")
    if selected_models == ["all"]:
        model_names = list(MODEL_CONFIGS)
    elif "all" in selected_models:
        raise ValueError("Informe 'all' como único valor de --models.")
    else:
        model_names = selected_models

    unknown = [name for name in model_names if name not in MODEL_CONFIGS]
    if unknown:
        known = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"Modelo(s) desconhecido(s): {', '.join(unknown)}; opções disponíveis: {known}")

    dataset_values = [_canonical_dataset_name(name) for name in _split_values(args.datasets)]
    if dataset_values == ["all"]:
        dataset_names = None
    elif "all" in dataset_values:
        raise ValueError("Informe 'all' como único valor de --datasets.")
    else:
        dataset_names = tuple(dataset_values) if dataset_values else None

    return [
        TrainingJob(
            model_name=model_name,
            config_path=_resolve_config_path(model_name),
            dataset_names=dataset_names,
        )
        for model_name in model_names
    ]


def _prompt_selection(prompt: str, options: Sequence[str]) -> tuple[str, ...]:
    """Solicita a seleção de vários itens por número ou nome no terminal."""
    option_names = tuple(options)
    while True:
        print()
        for index, option in enumerate(option_names, start=1):
            print(f"  {index}. {option}")
        try:
            raw = input(f"{prompt} (números, nomes ou 'all'): ").strip()
        except EOFError as exc:
            raise ValueError("A seleção interativa exige entrada pelo terminal.") from exc

        if not raw:
            print("Selecione pelo menos uma opção.")
            continue
        if raw.lower() == "all":
            return option_names

        selected: list[str] = []
        invalid: list[str] = []
        for value in _split_values([raw]):
            if value.isdigit():
                option_index = int(value) - 1
                if 0 <= option_index < len(option_names):
                    option = option_names[option_index]
                else:
                    invalid.append(value)
                    continue
            else:
                option = next(
                    (name for name in option_names if name.casefold() == value.casefold()),
                    None,
                )
                if option is None:
                    invalid.append(value)
                    continue
            if option not in selected:
                selected.append(option)

        if selected and not invalid:
            return tuple(selected)
        print(f"Seleção inválida: {', '.join(invalid) or raw}")


def interactive_jobs() -> list[TrainingJob]:
    """Monta uma fila pelos menus do terminal quando nenhuma seleção é informada."""
    selected_models = _prompt_selection("Selecione o(s) modelo(s)", tuple(MODEL_CONFIGS))
    jobs: list[TrainingJob] = []
    for model_name in selected_models:
        config_path = _resolve_config_path(model_name)
        config = load_training_config(config_path)
        configured_datasets = _available_dataset_names(
            config,
            include_missing_openfwi=True,
        )
        selected_datasets = _prompt_selection(
            f"Selecione o(s) conjunto(s) de dados para {model_name}",
            configured_datasets,
        )
        jobs.append(
            TrainingJob(
                model_name=model_name,
                config_path=config_path,
                dataset_names=selected_datasets,
            )
        )
    return jobs


def _validate_datasets(
        config: dict[str, Any],
        dataset_names: Sequence[str],
) -> None:
    _validate_supported_dataset_names(dataset_names)
    missing = []
    for dataset_name in dataset_names:
        try:
            dataset_config = config_for_dataset(config, dataset_name)
            resolve_dataset_files(dataset_config["datasets_root"], dataset_name)
        except FileNotFoundError:
            missing.append(dataset_name)
    if missing:
        raise FileNotFoundError(
            f"Conjunto(s) de dados não encontrado(s) em {config['datasets_root']}: {', '.join(missing)}"
        )


def _validate_supported_dataset_names(dataset_names: Sequence[str]) -> None:
    unsupported = [name for name in dataset_names if name not in SUPPORTED_DATASET_NAMES]
    if unsupported:
        raise ValueError(
            f"Conjuntos de dados não compatíveis: {', '.join(unsupported)}. "
            f"Opções do OpenFWI: {', '.join(OPENFWI_DATASET_NAMES)}; "
            "Marmousi também está disponível."
        )


def _configure_sample_selection(
        run_config: dict[str, Any],
        dataset_name: str,
        args: argparse.Namespace,
) -> None:
    if args.sample_indices is not None:
        run_config["sample_indices"] = args.sample_indices
        return

    if dataset_name in OPENFWI_DATASET_NAMES:
        requested_count = min(
            OPENFWI_DEFAULT_SAMPLE_COUNT,
            args.max_samples if args.max_samples is not None else OPENFWI_DEFAULT_SAMPLE_COUNT,
        )
        sample_count = min(
            requested_count,
            openfwi_sample_count(run_config["datasets_root"], dataset_name),
        )
        run_config["sample_indices"] = list(range(sample_count))
    elif args.max_samples is not None:
        run_config["max_samples"] = args.max_samples


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def run_combination(
        combination: TrainingCombination,
        args: argparse.Namespace,
        queue_position: int,
        queue_size: int,
        progress_reporter: TotalETA | None = None,
) -> list[dict[str, Any]]:
    """Executa um modelo e cenário para um único conjunto de dados."""
    job = combination.job
    dataset_name = combination.dataset_name
    run_config, configured_indices = _configure_combination(combination, args)
    device = resolve_device(args.device or run_config.get("device", "cuda_if_available"))
    release_memory(device)

    print(
        f"\nFila {queue_position}/{queue_size}: modelo={job.model_name} "
        f"dispositivo={device} conjunto de dados={dataset_name}"
    )

    sample_indices = (
        list(combination.sample_indices)
        if combination.sample_indices is not None
        else configured_indices
    )
    print(f"{dataset_name}: {len(sample_indices)} amostra(s)")
    summaries = run_training_samples(
        job.model_name,
        run_config,
        device,
        sample_indices=sample_indices,
        force=args.force,
        resume=not args.no_resume and not args.force,
        progress_reporter=progress_reporter,
    )
    for summary in summaries:
        if not summary["skipped"]:
            _print_summary(summary)

    release_memory(device)
    return summaries


def list_models() -> None:
    for model_name, config_path in MODEL_CONFIGS.items():
        print(f"{model_name}: {config_path.relative_to(PROJECT_ROOT)}")


def list_datasets() -> None:
    for model_name in MODEL_CONFIGS:
        config_path = _resolve_config_path(model_name)
        config = load_training_config(config_path)
        names = _available_dataset_names(
            config,
            include_missing_openfwi=True,
        )
        print(f"{model_name}: {', '.join(names)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODELO",
        help=f"Modelos que serão treinados ({', '.join(MODEL_CONFIGS)}). Use 'all' sozinho para selecionar todos.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="CONJUNTO",
        help="Conjuntos de dados OpenFWI e/ou Marmousi de cada modelo. Use 'all' para selecionar os definidos na configuração.",
    )
    parser.add_argument(
        "--device",
        help="Substitui o dispositivo definido na configuração (por exemplo, cpu ou cuda).",
    )
    parser.add_argument(
        "--results-version",
        default="v1",
        metavar="VERSÃO",
        help="Versão usada para organizar registros e resultados (padrão: v1).",
    )
    parser.add_argument(
        "--noise",
        dest="noise_ratios",
        nargs="+",
        type=float,
        metavar="FATOR",
        help=(
            "Ruído gaussiano como múltiplo do desvio padrão sísmico. Use 0 para "
            "sem ruído e informe vários valores para selecionar diferentes condições."
        ),
    )
    parser.add_argument(
        "--epochs",
        dest="epoch_values",
        nargs="+",
        type=int,
        metavar="ÉPOCAS",
        help="Substitui o número de épocas; informe vários valores junto com --product.",
    )
    parser.add_argument(
        "--geometry",
        dest="receiver_geometries",
        nargs="+",
        choices=("surface", "surface_wells"),
        metavar="GEOMETRIA",
        help=(
            "Geometria dos receptores. Todos os conjuntos de dados aceitam surface e "
            "surface_wells. No OpenFWI, o simulador local sintetiza os traços "
            "a partir do modelo de velocidade."
        ),
    )
    parser.add_argument(
        "--product",
        action="store_true",
        help=(
            "Combina os valores de ruído, épocas e geometrias compatíveis "
            "para cada conjunto de dados selecionado."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limita o número de amostras por tarefa; no OpenFWI, o padrão continua sendo usar as duas primeiras.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        help=(
            "Índices explícitos das amostras de cada conjunto de dados na fila; separe os valores por vírgula. "
            "Substitui a seleção padrão das duas primeiras amostras do OpenFWI."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Executa mesmo quando existe um checkpoint concluído; inicia sem retomar.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Não retoma checkpoints parciais. Execuções concluídas continuam sendo ignoradas, exceto quando --force é usado.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Valida e exibe a fila sem treinar.")
    parser.add_argument("--list-models", action="store_true", help="Lista os modelos suportados e encerra.")
    parser.add_argument("--list-datasets", action="store_true", help="Lista os conjuntos de dados configurados e encerra.")
    args = parser.parse_args(argv)
    args.sample_indices = _split_values([str(value) for value in args.sample_indices]) if args.sample_indices else None
    if args.sample_indices is not None:
        try:
            args.sample_indices = [int(value) for value in args.sample_indices]
        except ValueError as exc:
            parser.error(f"Índice de amostra inválido: {exc}")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples deve ser maior que zero.")
    if args.noise_ratios is not None:
        if any(not math.isfinite(value) for value in args.noise_ratios):
            parser.error("--noise aceita apenas valores numéricos finitos.")
        if any(value < 0.0 for value in args.noise_ratios):
            parser.error("Os valores de --noise não podem ser negativos.")
        if len(set(args.noise_ratios)) != len(args.noise_ratios):
            parser.error("Não repita valores em --noise.")
    if args.epoch_values is not None:
        if any(value <= 0 for value in args.epoch_values):
            parser.error("--epochs deve conter apenas valores maiores que zero.")
        if len(set(args.epoch_values)) != len(args.epoch_values):
            parser.error("Não repita valores em --epochs.")
    if args.receiver_geometries is not None and len(set(args.receiver_geometries)) != len(args.receiver_geometries):
        parser.error("Não repita valores em --geometry.")
    return args


def _run_training_queue(
        combinations: Sequence[TrainingCombination],
        args: argparse.Namespace,
        total_epochs: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, str]], TotalETA]:
    progress = TotalETA(total_epochs)
    pending = deque((index, combination, 0) for index, combination in enumerate(combinations))
    all_summaries: list[dict[str, Any]] = []
    # Guarde os detalhes da falha sem manter os rastreamentos de pilha. Um
    # rastreamento mantém vivos os quadros da execução que falhou, incluindo
    # conjunto de dados, modelo, otimizador e tensores do dispositivo, até o fim da fila.
    failures: list[tuple[int, str]] = []
    completed_count = 0

    while pending:
        index, combination, retry_count = pending.popleft()
        try:
            summaries = run_combination(
                combination,
                args,
                index + 1,
                len(combinations),
                progress,
            )
        except Exception as exc:
            attempt_number = retry_count + 1
            if retry_count < MAX_RETRIES_PER_COMBINATION:
                next_retry_count = retry_count + 1
                retry_entry = (index, combination, next_retry_count)
                pending.appendleft(retry_entry)
                print(
                    f"A combinação {index + 1}/{len(combinations)} falhou na "
                    f"tentativa {attempt_number}: {type(exc).__name__}: {exc}; "
                    f"nova tentativa imediata "
                    f"(tentativa {next_retry_count}/{MAX_RETRIES_PER_COMBINATION})."
                )
            else:
                failure_message = f"{type(exc).__name__}: {exc}"
                print(
                    f"A combinação {index + 1}/{len(combinations)} falhou após "
                    f"{attempt_number} tentativa(s): {failure_message}"
                )
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                failures.append((index, failure_message))
            continue

        all_summaries.extend(summaries)
        completed_count += 1
        print(
            f"Concluídas {completed_count}/{len(combinations)} combinações; "
            f"previsão total de término: {progress.eta}."
        )

    return all_summaries, failures, progress


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        list_models()
    if args.list_datasets:
        list_datasets()
    if args.list_models or args.list_datasets:
        return 0

    try:
        has_selection = bool(args.models or args.datasets)
        if not has_selection and sys.stdin.isatty():
            jobs = interactive_jobs()
        else:
            jobs = build_jobs(args)
        scenarios = build_scenarios(args)
        _validate_scenario_compatibility(jobs, scenarios)
        combinations = _expand_dataset_combinations(jobs, scenarios)
        combinations = _remove_completed_checkpoints(combinations, args)
        planned_epochs, training_tasks = _planned_epoch_count(combinations, args)
        if args.dry_run:
            print(f"Combinações de conjuntos de dados e cenários: {len(combinations)}")
            print(f"Tarefas de treinamento: {training_tasks}; épocas planejadas: {planned_epochs}")
            for index, combination in enumerate(combinations, start=1):
                config = load_training_config(combination.job.config_path)
                _validate_datasets(config, (combination.dataset_name,))
                print(
                    f"  Combinação {index}/{len(combinations)}: "
                    f"modelo={combination.job.model_name}, conjunto de dados={combination.dataset_name}, "
                    f"{_describe_scenario(combination.scenario)}"
                )
            return 0

        ensure_output_dirs()
        print(
            f"Carga total: {training_tasks} tarefa(s) de treinamento, {planned_epochs} época(s) "
            f"em {len(combinations)} combinação(ões) de conjunto de dados e cenário; "
            "previsão total de término: calculando"
        )
        all_summaries, failures, total_eta = _run_training_queue(
            combinations,
            args,
            planned_epochs,
        )
        print(
            f"\nFinalizadas {len(combinations)} combinação(ões); "
            f"{len(all_summaries)} execução(ões) de amostra processada(s); "
            f"previsão de tempo restante: {total_eta.eta}; "
            f"{len(failures)} combinação(ões) falharam após as tentativas."
        )
        return 1 if failures else 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"erro: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
