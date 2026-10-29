import torch.nn as nn

from .u_mamba_block import UMambaBlock


class MambaDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
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
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.block(x)
        return self.pool(features), features
