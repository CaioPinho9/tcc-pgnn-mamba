from __future__ import annotations

import torch
import torch.nn as nn

from helpers.mamba import load_mamba


Mamba = load_mamba()

from .coord_position_embedding import CoordPositionEmbedding2D


class BiMambaLayer2D(nn.Module):
    """Camada Mamba bidirecional com sequência achatada e embeddings posicionais 2D. Não é VMamba nem SS2D."""

    def __init__(
            self,
            channels: int,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.0,
            use_pos_embed: bool = True,
            d_state: int = 16,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba-ssm é necessário para BiMambaLayer2D. "
                "Instale com: pip install mamba-ssm causal-conv1d"
            )

        self.pos_embed = (
            CoordPositionEmbedding2D(channels) if use_pos_embed else None
        )
        self.norm = nn.LayerNorm(channels)

        self.mamba_fwd = Mamba(
            d_model=channels,
            d_state=int(d_state),
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=channels,
            d_state=int(d_state),
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(channels * 2, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape

        seq = x.flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            seq = seq + self.pos_embed(x)
        seq = self.norm(seq)

        y_fwd = self.mamba_fwd(seq)

        seq_rev = torch.flip(seq, dims=[1])
        y_bwd = self.mamba_bwd(seq_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)

        y = y.transpose(1, 2).reshape(b, c, h, w)

        return y
