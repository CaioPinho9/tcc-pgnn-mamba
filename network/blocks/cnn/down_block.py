from torch import nn

from .sub_block import SubBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            SubBlock(in_channels, out_channels),
            SubBlock(out_channels, out_channels),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = self.convs(x)
        return self.pool(features), features
