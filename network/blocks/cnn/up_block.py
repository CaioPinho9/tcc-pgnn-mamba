from torch import nn

from .sub_block import SubBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_shape=None):
        super().__init__()
        self.up = nn.Upsample(
            size=final_shape,
            scale_factor=None if final_shape else 2,
            mode="bilinear",
            align_corners=False,
        )
        self.convs = nn.Sequential(
            SubBlock(in_channels, out_channels),
            SubBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.up(self.convs(x))
