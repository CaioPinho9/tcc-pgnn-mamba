from __future__ import annotations

import copy
import gc
import json
import math
from pathlib import Path

import optuna
import torch

from network.models import build_model
from data.dataset import configured_sample_indices, prepare_data
from network.pipeline import train_model
from network.training_runner import build_hyperparameters as build_training_hyperparameters
from utils import seed_everything

from .common import (
    append_optuna_summary,
    compute_history_objective,
    CUDA_OOM_INVALID_REASON,
    history_last,
    history_min,
    INVALID_TRIAL_REASON_ATTR,
    stringify_for_csv,
)


def make_trial_config(base_config: dict, dataset_name: str, search_epochs: int, save_checkpoints: bool = True) -> dict:
    trial_config = copy.deepcopy(base_config)
    trial_config["dataset_name"] = dataset_name
    training = trial_config["training"]

    training["epochs"] = search_epochs
    training["verbose"] = False
    training["measure_runtime"] = False
    training["save_training_log"] = False
    training["save_checkpoints"] = save_checkpoints

    return trial_config


def make_hyperparameters(
        dataset_name: str,
        data: dict,
        params: dict,
        t_cfg: dict,
        sample_index: int,
        model_type: str,
        base_model_name: str,
) -> dict:
    run_config = {
        "dataset_name": dataset_name,
        "training": t_cfg,
    }
    hyperparameters = build_training_hyperparameters(
        model_type,
        run_config,
        data,
        sample_index=sample_index,
    )
    hyperparameters["model_name"] = base_model_name
    hyperparameters["trial_params"] = params
    return hyperparameters


def print_trial_params(params: dict) -> None:
    groups = {
        "training": (
            "learning_rate",
            "adam_betas",
            "lr_gamma",
            "lr_milestone",
            "gradient_clip_norm",
        ),
    }
    grouped_keys = {key for keys in groups.values() for key in keys}
    grouped = {
        group_name: {
            key: stringify_for_csv(params[key])
            for key in keys
            if key in params
        }
        for group_name, keys in groups.items()
    }
    model_params = {
        key: stringify_for_csv(value)
        for key, value in params.items()
        if key not in grouped_keys
    }
    if model_params:
        grouped["model"] = model_params

    print(json.dumps(grouped, indent=2, sort_keys=True), flush=True)


def make_epoch_callback(
        trial: optuna.Trial,
        step_offset: int,
        pruning_start_epoch: int,
        pruning_interval_epochs: int,
        history: list[dict] | None = None,
        metric: str = "model_loss",
        smoothing_window: int = 10,
):
    def epoch_callback(log_row: dict):
        if history is not None:
            history.append(dict(log_row))

        epoch = int(log_row["epoch"])
        if epoch < pruning_start_epoch or epoch % pruning_interval_epochs != 0:
            return

        if history is None:
            metric_values = [float(log_row[metric])]
        else:
            metric_values = [
                float(row[metric])
                for row in history[-smoothing_window:]
                if metric in row
            ]
        finite_values = [value for value in metric_values if math.isfinite(value)]
        if not finite_values:
            return

        smoothed_metric = sum(finite_values) / len(finite_values)
        trial.report(smoothed_metric, step=epoch + step_offset)

        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Podado na época {epoch} com {metric}={smoothed_metric:.6f} "
                f"(média_móvel_{len(finite_values)})"
            )

    return epoch_callback


def create_objective(
        suggest_params_fn,
        validation_dataset_names: list[str],
        config: dict,
        device,
        model_type: str,
        model_name: str,
        base_model_name: str,
        search_epochs: int,
        objective_metric: str = "model_loss",
        objective_aggregate: str = "mean_last_10",
        pruning_start_epoch: int = 500,
        pruning_interval_epochs: int = 50,
        optuna_summary_path: Path = None,
        data_by_dataset: dict = None,
        save_checkpoints: bool = True,
):
    cache_data = data_by_dataset is not None
    if data_by_dataset is None:
        data_by_dataset = {}
    planned_runs_per_trial = sum(
        len(configured_sample_indices(make_trial_config(config, dataset_name, search_epochs)))
        for dataset_name in validation_dataset_names
    )
    planned_epochs_per_trial = planned_runs_per_trial * search_epochs

    def set_epoch_budget_attrs(trial: optuna.Trial, completed_epochs: int, epoch_time_seconds: float) -> None:
        completed_epochs = min(int(completed_epochs), planned_epochs_per_trial)
        pruned_epochs = max(0, planned_epochs_per_trial - completed_epochs)
        trial.set_user_attr("optuna_epochs_completed", completed_epochs)
        trial.set_user_attr("optuna_planned_epochs", planned_epochs_per_trial)
        trial.set_user_attr("optuna_pruned_epochs", pruned_epochs)
        trial.set_user_attr("optuna_epoch_time_seconds", float(epoch_time_seconds))

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("selection_split", "validation")
        trial.set_user_attr("validation_dataset_names", list(validation_dataset_names))
        params = suggest_params_fn(trial)
        # A mesma combinação de parâmetros é avaliada nas amostras selecionadas para validação.
        dataset_losses: list[float] = []
        sample_run_index = 0
        completed_epochs = 0
        epoch_time_seconds = 0.0

        print_trial_params(params)

        for dataset_name in validation_dataset_names:
            run_config = make_trial_config(config, dataset_name, search_epochs, save_checkpoints=save_checkpoints)
            # A cópia recebe os valores do trial antes de construir modelo e dados para o treino.
            run_config["training"].update(copy.deepcopy(params))
            seed_everything(run_config.get("seed", 42))

            sample_indices = [int(value) for value in configured_sample_indices(run_config)]
            if not sample_indices:
                raise optuna.TrialPruned(f"Nenhuma amostra selecionada para o dataset {dataset_name}.")

            for sample_index in sample_indices:
                cache_key = (dataset_name, sample_index)
                if cache_data:
                    if cache_key not in data_by_dataset:
                        data_by_dataset[cache_key] = prepare_data(run_config, device, sample_index=sample_index)
                    data = data_by_dataset[cache_key]
                else:
                    data = prepare_data(run_config, device, sample_index=sample_index)

                hyperparameters = make_hyperparameters(
                    dataset_name=dataset_name,
                    data=data,
                    params=params,
                    t_cfg=run_config["training"],
                    sample_index=sample_index,
                    model_type=model_type,
                    base_model_name=base_model_name,
                )

                model = build_model(model_type, hyperparameters, device)
                run_name = f"{model_name}_optuna_{dataset_name}_s{sample_index:04d}_t{trial.number:04d}"
                checkpoint_prefix = f"hyperparameter_search/{run_name}"
                epoch_history: list[dict] = []
                sample_step_offset = search_epochs * sample_run_index

                print(f"{'=' * 70}")
                print(
                    f"Teste Optuna {trial.number:04d} | modelo: {model_type} "
                    f"| dataset: {dataset_name} | amostra: {sample_index}",
                    flush=True,
                )

                try:
                    # O treino prevê velocidade e compara sismogramas simulados com os observados.
                    summary = train_model(
                        run_name,
                        model,
                        hyperparameters,
                        run_config,
                        device,
                        checkpoint_prefix=checkpoint_prefix,
                        log_path=None,
                        data=data,
                        epoch_callback=make_epoch_callback(
                            trial,
                            sample_step_offset,
                            pruning_start_epoch,
                            pruning_interval_epochs,
                            history=epoch_history,
                            metric=objective_metric,
                        ),
                    )
                    completed_epochs += len(epoch_history)
                    epoch_time_seconds += sum(float(row.get("epoch_time_seconds", 0.0)) for row in epoch_history)
                    set_epoch_budget_attrs(trial, completed_epochs, epoch_time_seconds)

                    # Reduza o histórico das épocas a uma pontuação para esta amostra.
                    objective_loss, objective_epoch = compute_history_objective(
                        history=epoch_history,
                        metric=objective_metric,
                        aggregate=objective_aggregate,
                    )
                    dataset_losses.append(objective_loss)
                    print(
                        f"Teste Optuna {trial.number:04d} concluído | dataset: {dataset_name} "
                        f"| amostra: {sample_index} | {objective_metric}={objective_loss:.6g} "
                        f"| época_objetivo={objective_epoch}",
                        flush=True,
                    )

                    row = {
                        "trial_number": trial.number,
                        "dataset_name": dataset_name,
                        "sample_index": sample_index,
                        "objective_model_loss": objective_loss,
                        "objective_epoch": objective_epoch,
                        "objective_metric": objective_metric,
                        "objective_aggregate": objective_aggregate,
                        "selection_split": "validation",
                        "best_total_loss": float(summary.get("best_total_loss", float("nan"))),
                        "best_epoch": int(summary.get("best_epoch", -1)),
                        "min_model_loss": history_min(epoch_history, "model_loss"),
                        "min_data_loss": history_min(epoch_history, "data_loss"),
                        "final_data_loss": history_last(epoch_history, "data_loss"),
                        "final_model_loss": history_last(epoch_history, "model_loss"),
                        "checkpoint_path": summary.get("checkpoint_path", None),
                        **{k: stringify_for_csv(v) for k, v in params.items()},
                    }
                    append_optuna_summary(optuna_summary_path, row)

                    attr_prefix = f"{dataset_name}_s{sample_index:04d}"
                    trial.set_user_attr(f"{attr_prefix}_objective_model_loss", objective_loss)
                    trial.set_user_attr(f"{attr_prefix}_objective_epoch", objective_epoch)
                    trial.set_user_attr(f"{attr_prefix}_checkpoint_path", summary.get("checkpoint_path", None))
                    trial.set_user_attr(
                        f"{attr_prefix}_best_total_loss",
                        float(summary.get("best_total_loss", float("nan"))),
                    )

                except optuna.TrialPruned:
                    completed_epochs += len(epoch_history)
                    epoch_time_seconds += sum(float(row.get("epoch_time_seconds", 0.0)) for row in epoch_history)
                    set_epoch_budget_attrs(trial, completed_epochs, epoch_time_seconds)
                    raise

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("Memória CUDA esgotada. Teste podado como inválido.")
                        trial.set_user_attr(INVALID_TRIAL_REASON_ATTR, CUDA_OOM_INVALID_REASON)
                        set_epoch_budget_attrs(trial, completed_epochs, epoch_time_seconds)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        raise optuna.TrialPruned("Memória CUDA esgotada") from e
                    raise

                finally:
                    del model
                    if not cache_data:
                        del data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                sample_run_index += 1

        if not dataset_losses:
            raise optuna.TrialPruned("Nenhuma perda válida de dataset foi produzida.")

        # O valor do trial resume as execuções nas amostras e datasets de validação.
        return float(sum(dataset_losses) / len(dataset_losses))

    return objective
