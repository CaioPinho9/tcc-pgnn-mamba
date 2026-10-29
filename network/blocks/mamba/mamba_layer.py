from __future__ import annotations

import torch
import torch.nn as nn

from helpers.mamba import load_mamba


Mamba = load_mamba()


class MambaLayer2D(nn.Module):
    """Adaptador de mapa de características 2D para mamba_ssm.Mamba.

        Entrada e saída:
            (B, C, H, W)
    """

    def __init__(
            self,
            channels: int,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.0,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba-ssm é necessário para MambaLayer2D. "
                "Instale com: pip install mamba-ssm causal-conv1d"
            )

        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)

        return x
