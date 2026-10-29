import torch
import torch.nn as nn


class CoordPositionEmbedding2D(nn.Module):
    """Projeta coordenadas 2D normalizadas em embeddings por canal para mapas de
        características achatados. A entrada tem formato (B, C, H, W) e a saída,
        (B, H*W, C).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(2, channels)

    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((yy, xx), dim=-1).reshape(1, h * w, 2)
        pos = self.proj(coords)
        return pos.expand(b, -1, -1)
