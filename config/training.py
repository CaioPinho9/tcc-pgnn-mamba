from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingOptions:
    batch_size: int
    seed: int
    verbose: bool
    save_training_log: bool
    save_checkpoints: bool
    measure_runtime: bool
    evaluate_model_loss: bool
    report_freq: int

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        log_path: str | Path | None,
    ) -> "TrainingOptions":
        training = config["training"]
        dataloader = config.get("dataloader") or {}
        report_freq = int(training.get("report_freq", 1))
        return cls(
            batch_size=int(dataloader.get("batch_size", 4)),
            seed=int(config.get("seed", 42)),
            verbose=bool(training.get("verbose", True)),
            save_training_log=bool(training.get("save_training_log", True)) and log_path is not None,
            save_checkpoints=bool(training.get("save_checkpoints", True)),
            measure_runtime=bool(training.get("measure_runtime", True)),
            evaluate_model_loss=bool(training.get("evaluate_model_loss", True)),
            report_freq=max(1, report_freq),
        )
