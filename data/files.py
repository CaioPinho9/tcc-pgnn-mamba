from pathlib import Path

from config.config import DEFAULT_ARTIFACTS_ROOT

import numpy as np


def _natural_sort_key(path: Path) -> tuple:
    parts = path.stem.replace("-", "_").split("_")
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _is_readable_npy(path: Path) -> bool:
    try:
        np.load(path, mmap_mode="r")
        return True
    except (OSError, ValueError, EOFError):
        return False


def _first_readable(paths: list[Path], description: str) -> Path:
    for path in sorted(paths, key=_natural_sort_key):
        if _is_readable_npy(path):
            return path

    raise FileNotFoundError(f"Nenhum arquivo legível de {description} foi encontrado.")


def _first_valid_seis_vel_pair(dataset_dir: Path) -> tuple[Path, Path]:
    seismic_files = sorted(dataset_dir.glob("seis*.npy"), key=_natural_sort_key)
    velocity_files = {path.name: path for path in dataset_dir.glob("vel*.npy")}

    errors: list[str] = []

    for seismic_file in seismic_files:
        velocity_name = f"vel{seismic_file.name.removeprefix('seis')}"
        velocity_file = velocity_files.get(velocity_name)

        if velocity_file is None:
            errors.append(f"{seismic_file.name} -> ausente {velocity_name}")
            continue

        if not _is_readable_npy(seismic_file):
            errors.append(f"{seismic_file.name} -> ilegível")
            continue

        if not _is_readable_npy(velocity_file):
            errors.append(f"{velocity_file.name} -> ilegível")
            continue

        return seismic_file, velocity_file

    details = "; ".join(errors[:5])
    if len(errors) > 5:
        details += f"; ... and {len(errors) - 5} more"

    raise FileNotFoundError(
        f"Nenhum par legível de sísmica/velocidade foi encontrado em {dataset_dir}. "
        f"Foram verificados {len(seismic_files)} arquivos sísmicos. {details}"
    )


def resolve_dataset_files(datasets_root: str | Path, dataset_name: str) -> tuple[Path, Path]:
    dataset_dir = Path(datasets_root) / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"O dataset configurado não existe: {dataset_dir}")

    data_dir = dataset_dir / "data"
    model_dir = dataset_dir / "model"

    if data_dir.exists() and model_dir.exists():
        return (
            _first_readable(list(data_dir.glob("data*.npy")), "data"),
            _first_readable(list(model_dir.glob("model*.npy")), "model"),
        )

    if not list(dataset_dir.glob("seis*.npy")) or not list(dataset_dir.glob("vel*.npy")):
        raise FileNotFoundError(
            f"O dataset deve conter pastas data/model ou arquivos seis/vel correspondentes: {dataset_dir}"
        )

    return _first_valid_seis_vel_pair(dataset_dir)


def load_openfwi_arrays(
        dataset_root: str | Path,
        dataset_name: str,
        max_samples: int | None = None,
        sample_indices: int | list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data_file, label_file = resolve_dataset_files(dataset_root, dataset_name)

    try:
        x_data = np.load(data_file, mmap_mode="r")
        y_data = np.load(label_file, mmap_mode="r")
        if sample_indices is not None:
            indices = np.atleast_1d(sample_indices).astype(int)
            x = x_data[indices].astype(np.float32)
            y = y_data[indices].astype(np.float32)
        else:
            x = x_data[:max_samples].astype(np.float32)
            y = y_data[:max_samples].astype(np.float32)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(
            f"Falha ao carregar os arrays OpenFWI de {data_file} e {label_file}. "
            "O shard selecionado provavelmente está truncado ou ilegível."
        ) from exc

    return x, y


def openfwi_sample_count(dataset_root: str | Path, dataset_name: str) -> int:
    data_file, _ = resolve_dataset_files(dataset_root, dataset_name)
    try:
        return int(np.load(data_file, mmap_mode="r").shape[0])
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Falha ao inspecionar as amostras OpenFWI de {data_file}.") from exc


def get_artifacts_dir(dirname: str | Path | None = None) -> Path:
    path = Path(dirname) if dirname is not None else DEFAULT_ARTIFACTS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path
