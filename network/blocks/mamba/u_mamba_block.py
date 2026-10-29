import torch.nn as nn

from network.blocks.cnn.sub_block import SubBlock

from .bi_mamba_layer import BiMambaLayer2D


class UMambaBlock(nn.Module):
    """Bloco no estilo U-Mamba:
        SubBlock -> SubBlock -> BiMambaLayer2D opcional
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_mamba: bool = True,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.0,
            use_pos_embed: bool = False,
            conv_kernel_size: int = 3,
            d_state: int = 16,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            SubBlock(in_channels, out_channels, kernel_size=conv_kernel_size),
            SubBlock(out_channels, out_channels, kernel_size=conv_kernel_size),
        )
        self.mamba = (
            BiMambaLayer2D(
                out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                use_pos_embed=use_pos_embed,
            )
            if use_mamba
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.mamba(x)
