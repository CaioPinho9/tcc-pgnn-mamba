from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

torch.set_flush_denormal(False)

from utils import seed_everything

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
PATHS_CONFIG_PATH = CONFIG_DIR / "paths.json"


def _resolve_configured_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_paths_config() -> dict:
    loaded = json.loads(PATHS_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"A configuração de caminhos deve conter um objeto JSON: {PATHS_CONFIG_PATH}")
    return loaded


def _resolve_path_group(group: str) -> dict[str, Path]:
    return {
        name: _resolve_configured_path(value)
        for name, value in PATHS_CONFIG[group].items()
    }


PATHS_CONFIG = _load_paths_config()
DEFAULT_PATHS = _resolve_path_group("defaults")
DIRECTORIES = _resolve_path_group("directories")
DEFAULT_DATASETS_ROOT = DEFAULT_PATHS["datasets_root"]
DEFAULT_CHECKPOINT_ROOT = DEFAULT_PATHS["checkpoint_root"]
DEFAULT_ARTIFACTS_ROOT = DEFAULT_PATHS["artifacts_root"]
TRAINING_LOGS_ROOT = DIRECTORIES["training_logs"]
EVALUATION_ROOT = DIRECTORIES["evaluation"]
FIGURES_ROOT = DIRECTORIES["figures"]
OPTUNA_ROOT = DIRECTORIES["hyperparameter_search"]
PHYSICS_CONFIG_DIR = CONFIG_DIR / "physics"

MODEL_NAMES = (
    "pgnn_mardan_2024",
    "pgann_mardan_2024",
    "unet",
    "mamba",
    "unetmamba",
)

MODEL_CONFIGS = {
    name: CONFIG_DIR / "models" / f"{name}_config.json"
    for name in MODEL_NAMES
}
DATASET_PHYSICS_CONFIGS = {
    "Marmousi": PHYSICS_CONFIG_DIR / "marmousi.json",
}
OPENFWI_CONFIG_PATH = PHYSICS_CONFIG_DIR / "openfwi.json"


def config_for_dataset(
        config: dict,
        dataset_name: str,
        *,
        model_name: str | None = None,
) -> dict:
    """Aplica a física do dataset e as configurações explícitas de treinamento específicas dele."""
    merged = deepcopy(config)
    merged["dataset_name"] = dataset_name
    physics_config_path = DATASET_PHYSICS_CONFIGS.get(dataset_name)
    merged["physics_config"] = str(physics_config_path or OPENFWI_CONFIG_PATH)
    merged.pop("physics", None)
    if model_name is not None:
        model_datasets_root = PATHS_CONFIG.get("models", {}).get(model_name, {}).get("datasets_root")
        if model_datasets_root is not None:
            merged.setdefault("datasets_root", str(_resolve_configured_path(model_datasets_root)))
    dataset_root = PATHS_CONFIG.get("dataset_roots", {}).get(dataset_name)
    if dataset_root is not None:
        merged["datasets_root"] = str(_resolve_configured_path(dataset_root))
    dataset_training_by_name = merged.get("dataset_training", {})
    dataset_training = dataset_training_by_name.get(
        dataset_name,
        dataset_training_by_name.get("OpenFWI", {}) if physics_config_path is None else {},
    )
    merged.setdefault("training", {}).update(deepcopy(dataset_training))
    return merged


def load_openfwi_config(config_path: str | Path = OPENFWI_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração ausente: {path}")

    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"A configuração OpenFWI deve conter um objeto JSON: {path}")

    return loaded


OPENFWI_CONFIG = load_openfwi_config()


def load_physics_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        candidates = (
            Path(__file__).resolve().parent / path,
            Path(__file__).resolve().parents[1] / path,
            Path.cwd() / path,
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração física ausente: {path}")

    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"A configuração física deve conter um objeto JSON: {path}")
    return loaded


def resolve_physics_config(config: dict | None = None) -> dict:
    """Retorna a física padrão do OpenFWI com possíveis substituições por dataset."""
    physics_config_path = config.get("physics_config") if config else None
    resolved = deepcopy(
        load_physics_config(physics_config_path)
        if physics_config_path
        else OPENFWI_CONFIG
    )
    if config:
        resolved.update(config.get("physics", {}))
    return resolved


def load_training_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração ausente: {path}")
    loaded = json.loads(path.read_text())
    model_name = path.stem.removesuffix("_config")
    model_paths = PATHS_CONFIG.get("models", {}).get(model_name, {})
    loaded.setdefault(
        "datasets_root",
        str(_resolve_configured_path(model_paths.get("datasets_root", DEFAULT_DATASETS_ROOT))),
    )
    checkpointing = loaded.setdefault("checkpointing", {})
    checkpointing.setdefault(
        "dir",
        str(_resolve_configured_path(model_paths.get("checkpoint_root", DEFAULT_CHECKPOINT_ROOT))),
    )
    seed_everything(loaded.get("seed", 42))

    return loaded


def resolve_device(device_cfg: str = "cuda_if_available") -> torch.device:
    if device_cfg == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)
