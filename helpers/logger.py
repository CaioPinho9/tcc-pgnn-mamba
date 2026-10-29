from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


class TrainingLogger:
    TRAINING_LOG_COLUMNS = [
        "model_name", "epoch", "data_loss", "model_loss", "total_loss",
        "num_parameters", "neural_network_time_seconds",
        "physics_solver_time_seconds", "total_epoch_time_seconds",
        "epoch_time_seconds", "total_train_time_seconds",
        "peak_gpu_memory_mb", "gpu_hours", "learning_rate", "frequency_hz",
        "batch_size", "seed", "timestamp",
        "mean_inference_time_ms", "std_inference_time_ms", "num_inference_samples",
    ]

    def __init__(
        self,
        log_path: str | Path | None,
        *,
        model_name: str,
        total_epochs: int,
        start_epoch: int = 1,
        report_freq: int = 1,
        enabled: bool = True,
        verbose: bool = True,
        reset_log: bool = True,
        resume_epoch: int | None = None,
    ):
        self.log_path = Path(log_path) if log_path else None
        self.enabled = enabled and self.log_path is not None
        self.report_freq = max(1, int(report_freq))
        self.buffer: list[dict[str, Any]] = []

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if reset_log:
                self.log_path.unlink(missing_ok=True)
            elif resume_epoch is not None:
                self._truncate_to_epoch(resume_epoch)

        self.progress = tqdm(
            total=max(0, int(total_epochs)),
            initial=max(0, min(int(total_epochs), int(start_epoch) - 1)),
            desc=model_name,
            file=sys.stdout,
            disable=not verbose,
        )

    def _truncate_to_epoch(self, epoch: int) -> None:
        if not self.log_path.exists():
            return

        with self.log_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        kept = []
        for row in rows:
            try:
                row_epoch = int(row.get("epoch", -1))
            except (TypeError, ValueError):
                continue
            if row_epoch <= epoch:
                kept.append(row)
        if len(kept) == len(rows) and fieldnames == self.TRAINING_LOG_COLUMNS:
            return

        with self.log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.TRAINING_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {column: row.get(column, "") for column in self.TRAINING_LOG_COLUMNS}
                for row in kept
            )

    def log_epoch(self, row: dict[str, Any]) -> None:
        epoch = int(row.get("epoch", 0))
        if self.enabled:
            self.buffer.append(row)

        # Mantenha inalterada a frequência de atualização da barra tqdm e
        # atualize as métricas no intervalo de relatório configurado.
        if epoch == 1 or epoch >= self.progress.total or epoch % self.report_freq == 0:
            postfix = {
                "data_loss": row.get("data_loss", ""),
                "model_loss": row.get("model_loss", ""),
            }
            if "total_eta" in row:
                postfix["total_eta"] = row["total_eta"]
                postfix["total"] = row.get("total_epoch_progress", "")
            self.progress.set_postfix(postfix, refresh=False)

        delta = epoch - self.progress.n
        if delta > 0:
            self.progress.update(delta)

        if epoch == 1 or epoch >= self.progress.total or epoch % self.report_freq == 0:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self.buffer:
            return

        exists = self.log_path.exists() and self.log_path.stat().st_size > 0
        with self.log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.TRAINING_LOG_COLUMNS)
            if not exists:
                writer.writeheader()
            for row in self.buffer:
                writer.writerow({column: row.get(column, "") for column in self.TRAINING_LOG_COLUMNS})
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.progress.close()
