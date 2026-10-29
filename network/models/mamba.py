from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks.mamba.mamba_down_block import MambaDownBlock
from network.blocks.mamba.mamba_up_block import MambaUpBlock
from network.blocks.mamba.u_mamba_block import UMambaBlock


class MambaUNet(nn.Module):
    """Modelo direto:
        observação sísmica -> Mamba U-Net -> modelo de velocidade
    """

    def __init__(
            self,
            in_channels: int,
            n_blocks_encoder: int,
            n_blocks_decoder: int,
            final_spatial_shape: Tuple[int, int],
            base_channels: int = 16,
            final_out_channels: int = 1,
            use_mamba_encoder: bool = True,
            use_mamba_bottleneck: bool = True,
            use_mamba_decoder: bool = False,
            use_skip_connections: bool = False,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.0,
            use_pos_embed: bool = True,
            conv_kernel_size: int = 3,
            d_state: int = 16,
    ):
        super().__init__()
        if not 2 <= d_conv <= 4:
            raise ValueError(f"mamba d_conv deve estar entre 2 e 4 para causal_conv1d; obtido: {d_conv}")
        if int(d_state) <= 0:
            raise ValueError(f"mamba d_state deve ser positivo; obtido: {d_state}")
        conv_kernel_size = int(conv_kernel_size)
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(
                "mamba_conv_kernel_size deve ser um inteiro positivo ímpar"
            )

        n_blocks_encoder = int(n_blocks_encoder)
        n_blocks_decoder = int(n_blocks_decoder)
        if n_blocks_encoder <= 0 or n_blocks_decoder <= 0:
            raise ValueError("As quantidades de camadas do codificador e do decodificador Mamba devem ser positivas")

        if base_channels <= 0:
            raise ValueError("Os canais-base do Mamba devem ser positivos")

        encoder_channels = [base_channels * (2 ** i) for i in range(n_blocks_encoder + 1)]
        decoder_channels = [base_channels * (2 ** i) for i in range(n_blocks_decoder + 1)]
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.use_skip_connections = use_skip_connections

        self.enc_blocks = nn.ModuleList()
        for i in range(n_blocks_encoder):
            block_in_channels = in_channels if i == 0 else encoder_channels[i]
            self.enc_blocks.append(
                MambaDownBlock(
                    block_in_channels,
                    encoder_channels[i + 1],
                    use_mamba=use_mamba_encoder,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_pos_embed=use_pos_embed,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        bottleneck_ch = encoder_channels[-1]
        decoder_bottleneck_ch = decoder_channels[-1] * 2

        self.bottleneck = UMambaBlock(
            bottleneck_ch,
            decoder_bottleneck_ch,
            use_mamba=use_mamba_bottleneck,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            conv_kernel_size=conv_kernel_size,
        )

        self.dec_blocks = nn.ModuleList()
        current_ch = decoder_bottleneck_ch

        for stage_index in range(n_blocks_decoder):
            decoder_level = n_blocks_decoder - 1 - stage_index
            skip_level = n_blocks_encoder - 1 - stage_index
            skip_ch = (
                encoder_channels[skip_level + 1]
                if use_skip_connections and skip_level >= 0
                else 0
            )
            out_ch = decoder_channels[decoder_level]

            self.dec_blocks.append(
                MambaUpBlock(
                    current_ch + skip_ch,
                    out_ch,
                    final_shape=final_spatial_shape if stage_index == n_blocks_decoder - 1 else None,
                    use_mamba=use_mamba_decoder,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_pos_embed=use_pos_embed,
                    conv_kernel_size=conv_kernel_size,
                )
            )

            current_ch = out_ch

        self.final = nn.Sequential(
            nn.Conv2d(
                current_ch,
                final_out_channels,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = []

        for block in self.enc_blocks:
            x, features = block(x)
            if self.use_skip_connections:
                skips.append(features)

        x = self.bottleneck(x)

        for stage_index, block in enumerate(self.dec_blocks):
            if self.use_skip_connections:
                skip_index = len(skips) - 1 - stage_index
                if 0 <= skip_index < len(skips):
                    skip = skips[skip_index]
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(
                            x,
                            size=skip.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.final(x)
