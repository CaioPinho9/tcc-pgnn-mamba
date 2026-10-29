from pathlib import Path

from config.config import (
    DEFAULT_CHECKPOINT_ROOT,
    EVALUATION_ROOT,
    FIGURES_ROOT,
    TRAINING_LOGS_ROOT,
)


def ensure_output_dirs() -> None:
    for directory in (
        DEFAULT_CHECKPOINT_ROOT,
        TRAINING_LOGS_ROOT,
        EVALUATION_ROOT,
        FIGURES_ROOT,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
