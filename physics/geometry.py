from __future__ import annotations

from typing import Any

import numpy as np


def build_acquisition_geometry(
        physics_config: dict[str, Any],
        model_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    geometry = physics_config.get("receiver_geometry")
    if geometry is None:
        return None
    if geometry not in {"surface", "surface_wells", "surface_and_wells"}:
        raise ValueError(f"Geometria de receptores não suportada: {geometry!r}")

    depth, width = (int(value) for value in model_shape)
    source_count = int(physics_config["source_count"])
    source_depth = int(physics_config.get("source_depth_index", 0))

    if "source_x_spacing_cells" in physics_config:
        source_x = (
            int(physics_config.get("source_x_start_cells", 0))
            + np.arange(source_count, dtype=np.int64)
            * int(physics_config["source_x_spacing_cells"])
        )
    else:
        source_margin = int(
            physics_config.get(
                "source_x_start_cells",
                physics_config.get("source_x_margin_cells", 4),
            )
        )
        if "source_x_end_cells" in physics_config:
            source_end = int(physics_config["source_x_end_cells"])
        elif "source_x_start_cells" in physics_config:
            source_end = width - 1
        else:
            source_end = width - source_margin
        source_x_rounding = str(physics_config.get("source_x_rounding", "nearest")).lower()
        if source_x_rounding == "floor":
            source_x = np.floor(
                np.linspace(
                    source_margin,
                    source_end,
                    source_count,
                    dtype=np.float32,
                )
            ).astype(np.int64)
        elif source_x_rounding in {"nearest", "round"}:
            source_x = np.rint(
                np.linspace(source_margin, source_end, source_count)
            ).astype(np.int64)
        else:
            raise ValueError(
                f"source_x_rounding não suportado={source_x_rounding!r}; "
                "escolha 'nearest' ou 'floor'"
            )
    source_indices = np.column_stack([
        source_x,
        np.full(source_count, source_depth, dtype=np.int64),
    ])

    if geometry == "surface":
        if "surface_receiver_x_margin_cells" in physics_config:
            surface_margin = int(physics_config["surface_receiver_x_margin_cells"])
            receiver_x = np.arange(surface_margin, width - surface_margin, dtype=np.int64)
            receiver_depth = int(physics_config.get("surface_receiver_depth_index", 0))
        else:
            receiver_count = int(physics_config["receiver_count"])
            receiver_x = (
                int(physics_config.get("receiver_x_start_cells", 0))
                + np.arange(receiver_count, dtype=np.int64)
                * int(physics_config.get("receiver_x_spacing_cells", 1))
            )
            receiver_depth = int(physics_config.get("receiver_depth_index", 0))
        receiver_indices = np.column_stack([
            receiver_x,
            np.full(receiver_x.size, receiver_depth, dtype=np.int64),
        ])
    else:
        surface_margin = int(physics_config.get("surface_receiver_x_margin_cells", 2))
        surface_depth = int(physics_config.get("surface_receiver_depth_index", 4))
        surface_x = np.arange(surface_margin, width - surface_margin, dtype=np.int64)
        surface_indices = np.column_stack([
            surface_x,
            np.full(surface_x.size, surface_depth, dtype=np.int64),
        ])

        well_x_margin = int(physics_config.get("well_receiver_x_margin_cells", 4))
        well_z_margin = int(physics_config.get("well_receiver_z_margin_cells", 2))
        well_z = np.arange(well_z_margin, depth - well_z_margin, dtype=np.int64)
        left_well = np.column_stack([
            np.full(well_z.size, well_x_margin, dtype=np.int64),
            well_z,
        ])
        right_well = np.column_stack([
            np.full(well_z.size, width - well_x_margin, dtype=np.int64),
            well_z,
        ])
        receiver_indices = np.concatenate([left_well, surface_indices, right_well], axis=0)

    if np.any(source_indices[:, 0] < 0) or np.any(source_indices[:, 0] >= width):
        raise ValueError("A geometria das fontes ultrapassa a largura do modelo de velocidade")
    if np.any(source_indices[:, 1] < 0) or np.any(source_indices[:, 1] >= depth):
        raise ValueError("A geometria das fontes ultrapassa a profundidade do modelo de velocidade")
    if np.any(receiver_indices[:, 0] < 0) or np.any(receiver_indices[:, 0] >= width):
        raise ValueError("A geometria dos receptores ultrapassa a largura do modelo de velocidade")
    if np.any(receiver_indices[:, 1] < 0) or np.any(receiver_indices[:, 1] >= depth):
        raise ValueError("A geometria dos receptores ultrapassa a profundidade do modelo de velocidade")

    return {
        "source_indices": source_indices,
        "receiver_indices": receiver_indices,
    }
