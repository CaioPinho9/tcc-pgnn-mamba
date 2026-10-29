from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn

from .u_mamba_block import UMambaBlock


class MambaUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            final_shape: Optional[Tuple[int, int]] = None,
            use_mamba: bool = True,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.0,
            use_pos_embed: bool = True,
            conv_kernel_size: int = 3,
            d_state: int = 16,
    ):
        super().__init__()
        self.block = UMambaBlock(
            in_channels,
            out_channels,
            use_mamba=use_mamba,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            conv_kernel_size=conv_kernel_size,
        )
        self.up = nn.Upsample(
            size=final_shape,
            scale_factor=None if final_shape else 2,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        return self.up(self.block(x))
