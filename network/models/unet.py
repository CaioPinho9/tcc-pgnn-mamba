import torch
from torch import nn
import torch.nn.functional as F

from network.blocks.cnn.down_block import DownBlock
from network.blocks.cnn.sub_block import SubBlock
from network.blocks.cnn.up_block import UpBlock


class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            n_blocks_encoder,
            n_blocks_decoder,
            final_spatial_shape,
            final_out_channels=1,
    ):
        super().__init__()
        n_blocks_encoder = int(n_blocks_encoder)
        n_blocks_decoder = int(n_blocks_decoder)
        if n_blocks_decoder < 1 or n_blocks_decoder > n_blocks_encoder:
            raise ValueError(
                "n_blocks_decoder deve estar entre 1 e n_blocks_encoder"
            )
        self.out_channels = [2 ** i for i in range(n_blocks_encoder + 1)]

        self.enc_blocks = nn.ModuleList([
            DownBlock(self.out_channels[i] * in_channels, self.out_channels[i + 1] * in_channels)
            for i in range(n_blocks_encoder)
        ])

        bottleneck_ch = self.out_channels[-1] * in_channels
        self.bottleneck = nn.Sequential(
            SubBlock(bottleneck_ch, bottleneck_ch * 2),
            SubBlock(bottleneck_ch * 2, bottleneck_ch * 2),
        )

        self.dec_blocks = nn.ModuleList()
        current_ch = bottleneck_ch * 2
        for stage_index in range(n_blocks_decoder):
            skip_level = n_blocks_encoder - 1 - stage_index
            skip_ch = self.out_channels[skip_level + 1] * in_channels
            decoder_level = n_blocks_decoder - 1 - stage_index
            out_ch = self.out_channels[decoder_level] * in_channels
            self.dec_blocks.append(
                UpBlock(
                    current_ch + skip_ch,
                    out_ch,
                    final_shape=(
                        final_spatial_shape
                        if stage_index == n_blocks_decoder - 1
                        else None
                    ),
                )
            )
            current_ch = out_ch

        self.final = nn.Sequential(
            nn.Conv2d(current_ch, final_out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = []
        for block in self.enc_blocks:
            x, features = block(x)
            skips.append(features)

        x = self.bottleneck(x)

        for i, block in enumerate(self.dec_blocks):
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))

        return self.final(x)
