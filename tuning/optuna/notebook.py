from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import optuna
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_viz
from optuna.trial import TrialState

from config.config import OPTUNA_ROOT, config_for_dataset, load_training_config, resolve_device

from .common import INVALID_TRIAL_REASON_ATTR, counted_finished_trials, include_value
from .export import export_best_params
from .objective import create_objective

CONFIG_PATHS = {
    "pgnn_mardan_2024": "config/models/pgnn_mardan_2024_config.json",
    "pgann_mardan_2024": "config/models/pgann_mardan_2024_config.json",
    "unet": "config/models/unet_config.json",
    "unetmamba": "config/models/unetmamba_config.json",
    "mamba": "config/models/mamba_config.json",
}

MARMOUSI_DATASET_NAME = "Marmousi"
MARMOUSI_SEARCH_EPOCHS = 500
MAMBA_PATH_CANDIDATES = {
    "use_mamba_encoder": [False, True],
    "use_mamba_bottleneck": [False, True],
    "use_mamba_decoder": [False, True],
}
COMMON_SEARCH_PARAMS = {
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "lr_gamma",
    "lr_milestone",
    "gradient_clip_norm",
}
MAMBA_SEARCH_PARAMS = {
    "n_blocks_encoder",
    "n_blocks_decoder",
    "mamba_expand",
    "mamba_d_conv",
    "mamba_d_state",
    "mamba_base_channels",
    "mamba_dropout",
    "mamba_conv_kernel_size",
    "use_pos_embed",
    *MAMBA_PATH_CANDIDATES,
}
PGANN_SEARCH_PARAMS = {
    "pgann_encoder_channel_scale",
    "pgann_decoder_channel_scale",
    "pgann_attention_latent_kernel_size",
    "pgann_attention_feature_kernel_size",
    "pgann_attention_score_kernel_size",
    "pgann_attention_reduction",
}
MODEL_SEARCH_PARAMS = {
    "unet": COMMON_SEARCH_PARAMS,
    "mamba": COMMON_SEARCH_PARAMS | MAMBA_SEARCH_PARAMS,
    "unetmamba": COMMON_SEARCH_PARAMS | MAMBA_SEARCH_PARAMS,
    "pgann_mardan_2024": PGANN_SEARCH_PARAMS,
}


@dataclass
class OptunaNotebookConfig:
    model_type: str
    config_path: str
    model_name: str
    config: dict[str, Any]
    device: Any
    validation_dataset_names: list[str]
    max_trials: int
    search_seed: int
    objective_metric: str
    objective_aggregate: str
    search_params: tuple[str, ...]
    pruner_n_startup_trials: int = 10
    pruning_start_epoch: int = 1000
    pruning_interval_epochs: int = 1000
    optuna_study_version: str = "v16"
    categorical_values: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def training(self) -> dict[str, Any]:
        return self.config["training"]

    @property
    def is_marmousi(self) -> bool:
        return self.validation_dataset_names == [MARMOUSI_DATASET_NAME]

    @property
    def output_stem(self) -> str:
        if self.is_marmousi:
            return f"{self.model_type}_marmousi_optuna_500_fixed"
        return f"{self.model_type}_hparam_generic_optuna"

    @property
    def artifact_stem(self) -> str:
        return f"{self.output_stem}_{self.optuna_study_version}"

    @property
    def storage(self) -> str:
        return f"sqlite:///{OPTUNA_ROOT / f'{self.output_stem}.db'}"

    @property
    def study_name(self) -> str:
        datasets = "_".join(self.validation_dataset_names)
        return f"{self.model_name}_{datasets}_{self.optuna_study_version}"

    @property
    def search_epochs(self) -> int:
        return int(self.training.get("epochs", 2000))

    @property
    def base_adam_betas(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.training.get("adam_betas", [0.9, 0.999]))

def setup_optuna_notebook(
        model_type: str,
        *,
        validation_dataset_names: list[str],
        search_params: list[str] | tuple[str, ...],
        max_trials: int = 100,
        search_seed: int = 42,
        objective_metric: str = "model_loss",
        objective_aggregate: str = "mean_last_10",
        config_path: str | None = None,
        sample_indices: list[int] | None = None,
        datasets_root: str | None = None,
        optuna_study_version: str | None = None,
        categorical_values: dict[str, list[Any]] | None = None,
) -> OptunaNotebookConfig:
    if model_type not in CONFIG_PATHS:
        raise ValueError(f"MODEL_TYPE desconhecido: {model_type}. Escolha entre {list(CONFIG_PATHS)}")
    if not validation_dataset_names:
        raise ValueError("VALIDATION_DATASET_NAMES está vazio.")
    if MARMOUSI_DATASET_NAME in validation_dataset_names and validation_dataset_names != [MARMOUSI_DATASET_NAME]:
        raise ValueError("O Marmousi deve ser ajustado em um estudo separado do Optuna.")

    search_params = tuple(dict.fromkeys(str(name) for name in search_params))
    unknown_params = set(search_params) - MODEL_SEARCH_PARAMS[model_type]
    if unknown_params:
        raise ValueError(
            f"Parâmetros de busca não suportados para {model_type}: "
            f"{sorted(unknown_params)}"
        )
    categorical_values = {
        name: list(values) for name, values in (categorical_values or {}).items()
    }
    unsupported_choices = set(categorical_values) - (
        set(search_params) & (set(CATEGORICAL_VALUES) | {"lr_milestone"})
    )
    if unsupported_choices:
        raise ValueError(
            "Valores candidatos devem corresponder a parâmetros categóricos "
            f"selecionados na busca: {sorted(unsupported_choices)}"
        )
    for name, values in categorical_values.items():
        if not values:
            raise ValueError(f"A lista de candidatos de {name} está vazia.")
    config_path = config_path or CONFIG_PATHS[model_type]
    OPTUNA_ROOT.mkdir(parents=True, exist_ok=True)
    # Carrega a configuração-base antes de ajustá-la aos datasets de validação escolhidos.
    config = load_training_config(config_path)
    is_marmousi = validation_dataset_names == [MARMOUSI_DATASET_NAME]
    if is_marmousi:
        config = config_for_dataset(config, MARMOUSI_DATASET_NAME, model_name=model_type)
        if datasets_root is not None:
            resolved_datasets_root = str(Path(datasets_root).expanduser().resolve())
            config["datasets_root"] = resolved_datasets_root
        if sample_indices is not None:
            config["sample_indices"] = [int(value) for value in sample_indices]
        config["training"]["epochs"] = MARMOUSI_SEARCH_EPOCHS
    if model_type == "unet":
        mamba_config = load_training_config(CONFIG_PATHS["mamba"])
        if validation_dataset_names == [MARMOUSI_DATASET_NAME]:
            mamba_config = config_for_dataset(
                mamba_config,
                MARMOUSI_DATASET_NAME,
                model_name="mamba",
            )
        mamba_training = mamba_config["training"]
        config["training"]["n_blocks_encoder"] = int(
            mamba_training["n_blocks_encoder"]
        )
        config["training"]["n_blocks_decoder"] = int(
            mamba_training["n_blocks_decoder"]
        )
    device = resolve_device(config.get("device", "cuda_if_available"))
    if optuna_study_version is None:
        if validation_dataset_names == [MARMOUSI_DATASET_NAME]:
            optuna_study_version = "marmousi500_fixed_v1"
        else:
            optuna_study_version = "v16"
    notebook_config = OptunaNotebookConfig(
        model_type=model_type,
        config_path=config_path,
        model_name=(
            f"{model_type}_marmousi_500"
            if validation_dataset_names == [MARMOUSI_DATASET_NAME]
            else f"{model_type}_hparam_generic"
        ),
        config=config,
        device=device,
        validation_dataset_names=validation_dataset_names,
        max_trials=max_trials,
        search_seed=search_seed,
        objective_metric=objective_metric,
        objective_aggregate=objective_aggregate,
        search_params=search_params,
        pruner_n_startup_trials=2 if is_marmousi else 10,
        pruning_start_epoch=100 if is_marmousi else 1000,
        pruning_interval_epochs=50 if is_marmousi else 1000,
        optuna_study_version=optuna_study_version,
        categorical_values=categorical_values,
    )

    print("tipo de modelo:        ", notebook_config.model_type)
    print("caminho da configuração:", notebook_config.config_path)
    print("dispositivo:            ", notebook_config.device)
    print("datasets de validação:  ", ", ".join(notebook_config.validation_dataset_names))
    print("épocas de busca:        ", notebook_config.search_epochs)
    print("máximo de testes:       ", notebook_config.max_trials)
    return notebook_config

CATEGORICAL_VALUES = {
    "pgann_encoder_channel_scale": [0.5, 0.75, 1.0],
    "pgann_decoder_channel_scale": [0.5, 0.75, 1.0],
    "pgann_attention_latent_kernel_size": [1, 3, 5],
    "pgann_attention_feature_kernel_size": [1, 3, 5],
    "pgann_attention_score_kernel_size": [1, 3, 5],
    "pgann_attention_reduction": [1, 2, 4, 8],
    "mamba_expand": [1, 2, 4],
    "mamba_d_conv": [2, 3, 4],
    "mamba_d_state": [8, 16, 32],
    "mamba_base_channels": [16, 24, 32, 48],
    "mamba_dropout": [0.0, 0.05, 0.10, 0.20],
    "mamba_conv_kernel_size": [1, 3, 5],
    "use_pos_embed": [False, True],
    "gradient_clip_norm": [0.25, 0.5, 1.0, 2.0, 5.0],
    **MAMBA_PATH_CANDIDATES,
    "n_blocks_encoder": [3, 4, 5],
    "n_blocks_decoder": [2, 3, 4, 5],
    "encoder_base_channels": [16, 24, 32, 36, 40, 48, 64],
    "decoder_initial_channels": [48, 56, 64, 72, 80, 88, 96],
}
NUMERIC_RANGES = {
    "learning_rate": ((1e-5, 1e-2, None, False), (1e-4, 3e-3, None, True)),
    "lr_gamma": ((0.90, 1.0, 0.01, False), (0.90, 1.0, 0.01, False)),
    "adam_beta1": ((0.5, 0.99, 0.01, False), (0.50, 0.95, 0.01, False)),
    "adam_beta2": ((0.9, 0.999, 0.001, False), (0.900, 0.999, 0.001, False)),
}


def _base_value(training: dict[str, Any], name: str) -> Any:
    if name == "adam_beta1":
        return float(training.get("adam_betas", [0.9, 0.999])[0])
    if name == "adam_beta2":
        return float(training.get("adam_betas", [0.9, 0.999])[1])
    defaults = {
        "learning_rate": 1e-3,
        "lr_gamma": 1.0,
        "lr_milestone": 100,
        "gradient_clip_norm": 1.0,
        "mamba_expand": 2,
        "mamba_d_conv": 4,
        "mamba_d_state": 16,
        "mamba_base_channels": 16,
        "mamba_dropout": 0.0,
        "mamba_conv_kernel_size": 3,
        "pgann_encoder_channel_scale": 1.0,
        "pgann_decoder_channel_scale": 1.0,
        "pgann_attention_latent_kernel_size": 1,
        "pgann_attention_feature_kernel_size": 1,
        "pgann_attention_score_kernel_size": 1,
        "pgann_attention_reduction": 2,
        "use_skip_connections": False,
        "use_pos_embed": True,
        **{name: False for name in MAMBA_PATH_CANDIDATES},
        "n_blocks_encoder": 4,
        "n_blocks_decoder": 4,
    }
    return training.get(name, defaults.get(name))


def _choices(config: OptunaNotebookConfig, name: str) -> list:
    # Listas explícitas são usadas sem acrescentar valores da configuração-base.
    if name in config.categorical_values:
        return list(config.categorical_values[name])
    training = config.training
    if name == "lr_milestone":
        values = [50, 100, 200, 300, 400] if config.is_marmousi else [10, 50, 100, 200]
    else:
        values = CATEGORICAL_VALUES.get(name, [])
    current = _base_value(training, name)
    return include_value(values, current) if current is not None else values


def _suggest_value(
        trial: optuna.Trial,
        config: OptunaNotebookConfig,
        name: str,
        trial_name: str,
) -> Any:
    ranges = NUMERIC_RANGES.get(name)
    spec = ranges[1 if config.is_marmousi else 0] if ranges else None
    if spec is not None:
        low, high, step, log = spec
        current = _base_value(config.training, name)
        low = min(low, current)
        high = max(high, current)
        sampler = trial.suggest_float
        if isinstance(current, int) and not isinstance(current, bool):
            sampler = trial.suggest_int
        return sampler(trial_name, low, high, step=step, log=log)
    return trial.suggest_categorical(
        trial_name, _choices(config, name)
    )


def _trial_name(name: str, is_marmousi: bool) -> str:
    aliases = {
        "learning_rate": "learning_rate_v3",
        "lr_gamma": "lr_gamma_v2",
    }
    return name if is_marmousi else aliases.get(name, name)


def make_suggest_params(
        notebook_config: OptunaNotebookConfig,
) -> Callable[[optuna.Trial], dict[str, Any]]:
    training = notebook_config.training
    search_params = notebook_config.search_params

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        # Cada tentativa recebe valores para os parâmetros selecionados neste estudo.
        params: dict[str, Any] = {}
        betas = list(notebook_config.base_adam_betas)
        for name in search_params:
            trial_name = _trial_name(name, notebook_config.is_marmousi)
            if name in {"adam_beta1", "adam_beta2"}:
                index = 0 if name == "adam_beta1" else 1
                betas[index] = _suggest_value(
                    trial, notebook_config, name, trial_name
                )
            else:
                params[name] = _suggest_value(
                    trial, notebook_config, name, trial_name
                )

        if "adam_beta1" in search_params or "adam_beta2" in search_params:
            if betas[0] >= betas[1]:
                trial.set_user_attr(INVALID_TRIAL_REASON_ATTR, "invalid_adam_betas")
                raise optuna.TrialPruned("Invalid Adam beta order")
            params["adam_betas"] = tuple(betas)

        return params

    return suggest


def print_search_ranges(notebook_config: OptunaNotebookConfig) -> None:
    print("Parâmetros pesquisados:", list(notebook_config.search_params))
    for name in notebook_config.search_params:
        if name not in NUMERIC_RANGES:
            print(f"  {name}: {_choices(notebook_config, name)}")
    print("Configuração base/fixa:", json.dumps(notebook_config.training, sort_keys=True))


def make_pruner(notebook_config: OptunaNotebookConfig):
    if notebook_config.is_marmousi:
        return optuna.pruners.PercentilePruner(
            percentile=75.0,
            n_startup_trials=notebook_config.pruner_n_startup_trials,
            n_warmup_steps=notebook_config.pruning_start_epoch,
            interval_steps=notebook_config.pruning_interval_epochs,
            n_min_trials=2,
        )
    return optuna.pruners.MedianPruner(
        n_startup_trials=notebook_config.pruner_n_startup_trials,
        n_warmup_steps=notebook_config.pruning_start_epoch,
        interval_steps=notebook_config.pruning_interval_epochs,
    )


def _baseline_trial_params(notebook_config: OptunaNotebookConfig) -> dict[str, Any]:
    training = notebook_config.training
    baseline: dict[str, Any] = {}
    for name in notebook_config.search_params:
        value = _base_value(training, name)
        if name in notebook_config.categorical_values and value not in _choices(notebook_config, name):
            return {}
        if value is not None:
            baseline[_trial_name(name, notebook_config.is_marmousi)] = value

    return baseline


def create_or_load_study(notebook_config: OptunaNotebookConfig) -> tuple[optuna.Study, int]:
    sampler = optuna.samplers.TPESampler(seed=notebook_config.search_seed)
    pruner = make_pruner(notebook_config)
    study = optuna.create_study(
        study_name=notebook_config.study_name,
        storage=notebook_config.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # O Optuna exige candidatos categóricos constantes dentro do mesmo estudo.
    for previous_trial in study.trials:
        for name in notebook_config.categorical_values:
            trial_name = _trial_name(name, notebook_config.is_marmousi)
            distribution = previous_trial.distributions.get(trial_name)
            if distribution is not None:
                try:
                    optuna.distributions.check_distribution_compatibility(
                        distribution,
                        optuna.distributions.CategoricalDistribution(_choices(notebook_config, name)),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Os candidatos de {name} mudaram. Altere optuna_study_version "
                        "para iniciar um estudo separado com os novos valores."
                    ) from error
            fixed_params = previous_trial.system_attrs.get("fixed_params", {})
            if previous_trial.state == TrialState.WAITING and trial_name in fixed_params:
                if fixed_params[trial_name] not in _choices(notebook_config, name):
                    raise ValueError(
                        f"Há um teste na fila com {name} fora dos candidatos atuais. "
                        "Altere optuna_study_version para iniciar um estudo separado."
                    )

    if len(study.trials) == 0:
        enqueue_params = _baseline_trial_params(notebook_config)
        if enqueue_params:
            study.enqueue_trial(enqueue_params)
            print("Baseline do config selecionado e colocado como primeiro teste:")
            print(json.dumps(enqueue_params, indent=2))
        elif notebook_config.categorical_values:
            print("Baseline fora dos candidatos escolhidos; a busca usará somente as listas informadas.")
    else:
        print(f"O estudo já possui {len(study.trials)} teste(s); o baseline não será colocado na fila novamente.")

    finished_trials = counted_finished_trials(study.trials)
    remaining_trials = max(0, notebook_config.max_trials - len(finished_trials))
    print("nome do estudo:       ", notebook_config.study_name)
    print("armazenamento:        ", notebook_config.storage)
    print("testes existentes:    ", len(study.trials))
    print("testes finalizados:   ", len(finished_trials))
    print("testes restantes:     ", remaining_trials)
    print_epoch_budget_report(notebook_config, study, remaining_trials)
    return study, remaining_trials


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "indisponível"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _trial_completed_epochs(trial: optuna.trial.FrozenTrial, planned_epochs_per_trial: int) -> int:
    attr_value = trial.user_attrs.get("optuna_epochs_completed")
    if attr_value is not None:
        return min(int(attr_value), planned_epochs_per_trial)
    if trial.state == TrialState.COMPLETE:
        return planned_epochs_per_trial
    if trial.intermediate_values:
        return min(max(int(step) for step in trial.intermediate_values), planned_epochs_per_trial)
    return 0


def _observed_epoch_time(study: optuna.Study, planned_epochs_per_trial: int) -> tuple[float, int]:
    observed_seconds = 0.0
    observed_epochs = 0
    for trial in study.trials:
        seconds = trial.user_attrs.get("optuna_epoch_time_seconds")
        epochs = _trial_completed_epochs(trial, planned_epochs_per_trial)
        if seconds is None or epochs <= 0:
            continue
        observed_seconds += float(seconds)
        observed_epochs += epochs
    return observed_seconds, observed_epochs


def print_epoch_budget_report(
        notebook_config: OptunaNotebookConfig,
        study: optuna.Study,
        remaining_trials: int,
) -> None:
    planned_epochs_per_trial = len(notebook_config.validation_dataset_names) * notebook_config.search_epochs
    total_planned_epochs = notebook_config.max_trials * planned_epochs_per_trial
    finished_trials = counted_finished_trials(study.trials)
    completed_epochs = sum(
        _trial_completed_epochs(trial, planned_epochs_per_trial)
        for trial in finished_trials
    )
    finished_planned_epochs = len(finished_trials) * planned_epochs_per_trial
    unexecuted_epochs = max(0, finished_planned_epochs - completed_epochs)
    remaining_best_epochs = remaining_trials * min(notebook_config.pruning_start_epoch, planned_epochs_per_trial)
    remaining_worst_epochs = remaining_trials * planned_epochs_per_trial

    observed_seconds, observed_epochs = _observed_epoch_time(study, planned_epochs_per_trial)
    seconds_per_epoch = observed_seconds / observed_epochs if observed_epochs else None
    observed_epochs_per_trial = completed_epochs / len(finished_trials) if finished_trials else None
    if observed_epochs_per_trial is None:
        remaining_estimated_epochs = remaining_worst_epochs
    else:
        remaining_estimated_epochs = int(round(remaining_trials * observed_epochs_per_trial))

    consumed_pct = (completed_epochs / total_planned_epochs * 100.0) if total_planned_epochs else 0.0
    unexecuted_pct_finished = (unexecuted_epochs / finished_planned_epochs * 100.0) if finished_planned_epochs else 0.0
    unexecuted_pct_total = (unexecuted_epochs / total_planned_epochs * 100.0) if total_planned_epochs else 0.0

    print("orçamento de épocas:")
    print(f"  por teste:          {planned_epochs_per_trial:,} épocas")
    print(f"  total planejado:    {total_planned_epochs:,} épocas")
    print(f"  concluídas/executadas: {completed_epochs:,} épocas ({consumed_pct:.2f}% do total)")
    print(
        f"  não executadas:     {unexecuted_epochs:,} épocas "
        f"({unexecuted_pct_finished:.2f}% dos testes finalizados, {unexecuted_pct_total:.2f}% do total)"
    )
    print(f"  testes restantes:   {remaining_trials}")
    print(f"  melhor restante:    {remaining_best_epochs:,} épocas")
    print(f"  estimativa restante: {remaining_estimated_epochs:,} épocas")
    print(f"  pior restante:      {remaining_worst_epochs:,} épocas")
    if seconds_per_epoch is None:
        print("  cenários de tempo:  indisponíveis até concluir ao menos uma época cronometrada")
    else:
        print(f"  tempo médio/época:  {seconds_per_epoch:.2f}s")
        print(f"  melhor caso:        {_format_seconds(remaining_best_epochs * seconds_per_epoch)}")
        print(f"  tempo estimado:     {_format_seconds(remaining_estimated_epochs * seconds_per_epoch)}")
        print(f"  pior caso:          {_format_seconds(remaining_worst_epochs * seconds_per_epoch)}")


def make_epoch_budget_callback(notebook_config: OptunaNotebookConfig):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        finished_trials = counted_finished_trials(study.trials)
        remaining_trials = max(0, notebook_config.max_trials - len(finished_trials))
        print_epoch_budget_report(notebook_config, study, remaining_trials)

    return callback


def run_optuna_study(
        notebook_config: OptunaNotebookConfig,
        study: optuna.Study,
        remaining_trials: int,
) -> None:
    if remaining_trials > 0:
        optuna_summary_path = OPTUNA_ROOT / f"{notebook_config.artifact_stem}_summary.csv"
        # A função objetivo aplica os parâmetros sugeridos, treina na validação e retorna uma nota agregada.
        objective = create_objective(
            suggest_params_fn=make_suggest_params(notebook_config),
            validation_dataset_names=notebook_config.validation_dataset_names,
            config=notebook_config.config,
            device=notebook_config.device,
            model_type=notebook_config.model_type,
            model_name=notebook_config.model_name,
            base_model_name=notebook_config.model_type,
            search_epochs=notebook_config.search_epochs,
            objective_metric=notebook_config.objective_metric,
            objective_aggregate=notebook_config.objective_aggregate,
            pruning_start_epoch=notebook_config.pruning_start_epoch,
            pruning_interval_epochs=notebook_config.pruning_interval_epochs,
            optuna_summary_path=optuna_summary_path,
            save_checkpoints=False,
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[make_epoch_budget_callback(notebook_config)],
        )
    else:
        print("Não há testes restantes para executar. Aumente hparam_max_trials ou crie um novo nome de estudo.")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        print("Melhor valor:", study.best_value)
        print("Melhores parâmetros:")
        print(json.dumps(study.best_params, indent=2, sort_keys=True))
    else:
        print("Ainda não há testes concluídos.")


def display_trials(study: optuna.Study):
    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        return trials_df.sort_values("value", na_position="last").head(20)
    return trials_df


def _study_without_failed_trials(study: optuna.Study) -> optuna.Study:
    filtered_study = optuna.create_study(direction=study.direction)
    filtered_study.add_trials(
        [trial for trial in study.trials if trial.state != TrialState.FAIL]
    )
    return filtered_study


def print_param_importances(study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) >= 2:
        importances = optuna.importance.get_param_importances(study)
        print(importances)
    else:
        print("São necessários pelo menos 2 testes concluídos para uma importância de parâmetros confiável.")


def plot_param_importances(notebook_config: OptunaNotebookConfig, study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < 2:
        print("São necessários pelo menos 2 testes concluídos para plotar a importância dos parâmetros.")
        return

    optuna_viz.plot_param_importances(_study_without_failed_trials(study))
    plt.tight_layout()
    plt.savefig(
        OPTUNA_ROOT / f"{notebook_config.artifact_stem}_param_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def plot_optimization_history(notebook_config: OptunaNotebookConfig, study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < 2:
        print("São necessários pelo menos 2 testes concluídos para plotar o histórico de otimização.")
        return

    optuna_viz.plot_optimization_history(_study_without_failed_trials(study))
    plt.tight_layout()
    plt.savefig(
        OPTUNA_ROOT / f"{notebook_config.artifact_stem}_optimization_history.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def export_best_trial_params(notebook_config: OptunaNotebookConfig, study: optuna.Study) -> None:
    best_params_path = OPTUNA_ROOT / f"{notebook_config.artifact_stem}_best_params.json"
    export_best_params(
        study=study,
        best_params_path=best_params_path,
        training_params=notebook_config.training,
        objective_metric=notebook_config.objective_metric,
        objective_aggregate=notebook_config.objective_aggregate,
        epochs=notebook_config.search_epochs,
    )
