from __future__ import annotations

import json
from pathlib import Path

import optuna


def _best_param(best_params: dict, key: str, default):
    if key in best_params:
        return best_params[key]

    versioned_keys = sorted(
        candidate_key
        for candidate_key in best_params
        if candidate_key.startswith(f"{key}_v")
    )
    if versioned_keys:
        return best_params[versioned_keys[-1]]

    return default


def export_best_params(
        study,
        best_params_path: Path,
        training_params: dict,
        objective_metric: str,
        objective_aggregate: str,
        epochs: int = 2000,
) -> dict | None:
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("Ainda não há testes concluídos.")
        return None

    best_trial = study.best_trial
    # Combine a configuração de treino-base com os parâmetros selecionados pelo estudo.
    best_params = dict(training_params)
    best_params.update(study.best_params)

    base_betas = training_params.get("adam_betas", (0.9, 0.999))
    best_params["learning_rate"] = float(
        _best_param(best_params, "learning_rate", 1e-3)
    )
    best_params["lr_gamma"] = float(_best_param(best_params, "lr_gamma", 1.0))
    best_params["adam_betas"] = (
        float(_best_param(best_params, "adam_beta1", base_betas[0])),
        float(_best_param(best_params, "adam_beta2", base_betas[1])),
    )
    best_params["gradient_clip_norm"] = float(
        best_params.get("gradient_clip_norm", 1.0)
    )
    best_params["epochs"] = int(epochs)
    best_params["objective_metric"] = objective_metric
    best_params["objective_aggregate"] = objective_aggregate
    best_params["selection_split"] = "validation"
    best_params["best_trial_number"] = best_trial.number
    best_params["best_trial_value"] = float(best_trial.value)
    best_params["note"] = (
        f"A seleção do Optuna usa a validação agregada {objective_metric}; "
        "avalie a configuração congelada uma vez no conjunto de teste."
    )

    serializable_best_params = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in best_params.items()
    }
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(
        json.dumps(serializable_best_params, indent=2, sort_keys=True)
    )

    print(f"Salvo em: {best_params_path}")
    print("\nMelhor teste do Optuna:")
    print("  número do teste: ", best_trial.number)
    print("  valor:            ", best_trial.value)
    print("  épocas:           ", epochs)
    print("\nJSON dos melhores parâmetros:")
    print(json.dumps(serializable_best_params, indent=2, sort_keys=True))

    return serializable_best_params
