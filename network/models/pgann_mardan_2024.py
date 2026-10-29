from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .pinn_fwi.networks import Block


class _LatentGuidedAttention(nn.Module):
    """Aplica um mapa de atenção espacial a um mapa de características do decodificador."""

    def __init__(
            self,
            latent_channels: int,
            feature_channels: int,
            projection_channels: int,
            latent_kernel_size: int,
            feature_kernel_size: int,
            score_kernel_size: int,
    ) -> None:
        super().__init__()
        self.latent_projection = nn.Conv2d(
            latent_channels,
            projection_channels,
            kernel_size=latent_kernel_size,
            padding=latent_kernel_size // 2,
        )
        self.feature_projection = nn.Conv2d(
            feature_channels,
            projection_channels,
            kernel_size=feature_kernel_size,
            padding=feature_kernel_size // 2,
        )
        self.attention_projection = nn.Conv2d(
            projection_channels,
            1,
            kernel_size=score_kernel_size,
            padding=score_kernel_size // 2,
        )

    def forward(
            self,
            latent: torch.Tensor,
            feature: torch.Tensor,
            output_shape: tuple[int, int],
    ) -> torch.Tensor:
        latent_projection = self.latent_projection(latent)
        latent_projection = F.interpolate(
            latent_projection,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_projection = self.feature_projection(feature)
        attention_logits = self.attention_projection(
            F.relu(latent_projection + feature_projection)
        )
        batch, _, height, width = attention_logits.shape
        attention_map = torch.softmax(
            attention_logits.reshape(batch, 1, height * width), dim=-1
        ).reshape(batch, 1, height, width)
        # Alinha o mapa à resolução de l antes da multiplicação, como no esquema.
        attention_map = F.interpolate(
            attention_map,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attended = feature * attention_map
        # O artigo redimensiona o dado ponderado para a resolução da saída.
        return F.interpolate(
            attended,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )


class PGANNMardan2024(nn.Module):
    """Reconstrução local da topologia PGANN apresentada na Figura 1 de Mardan.

    O codificador reutiliza Block da PGNN e preserva o latente espacial, sem
    camada totalmente conectada. Do latente partem o decodificador principal e
    um caminho separado de três blocos que fornece características aos operadores
    de atenção guiados pelo latente. As saídas ponderadas são concatenadas à saída
    totalmente decodificada.

    A figura mostra os dois caminhos, mas não especifica o compartilhamento de
    pesos; esta implementação usa parâmetros independentes. As larguras de canais,
    os kernels de atenção, a redução das projeções, o eixo do Softmax e os detalhes
    de interpolação também são escolhas locais.
    Os valores padrão preservam a implementação com canais crescentes no
    codificador, decrescentes no decodificador e projeções de atenção 1x1.
    """

    def __init__(
            self,
            in_channels: int,
            n_blocks_encoder: int,
            n_blocks_decoder: int,
            final_spatial_shape: tuple[int, int],
            m_min: float = 0.0,
            m_max: float = 1.0,
            final_out_channels: int = 1,
            encoder_channel_scale: float = 1.0,
            decoder_channel_scale: float = 1.0,
            attention_latent_kernel_size: int = 1,
            attention_feature_kernel_size: int = 1,
            attention_score_kernel_size: int = 1,
            attention_reduction: int = 2,
    ) -> None:
        super().__init__()
        if n_blocks_encoder < 1:
            raise ValueError("n_blocks_encoder deve ser pelo menos 1")
        if n_blocks_decoder < 4:
            raise ValueError(
                "O PGANN requer pelo menos quatro blocos no decodificador principal"
            )
        if any(
            not math.isfinite(scale) or scale <= 0
            for scale in (encoder_channel_scale, decoder_channel_scale)
        ):
            raise ValueError("As escalas de canais devem ser positivas e finitas")
        attention_kernel_sizes = (
            attention_latent_kernel_size,
            attention_feature_kernel_size,
            attention_score_kernel_size,
        )
        if any(type(size) is not int or size < 1 or size % 2 == 0 for size in attention_kernel_sizes):
            raise ValueError("Os kernels de atenção devem ser inteiros positivos ímpares")
        if type(attention_reduction) is not int or attention_reduction < 1:
            raise ValueError("attention_reduction deve ser um inteiro positivo")

        self.in_channels = int(in_channels)
        self.final_spatial_shape = tuple(int(value) for value in final_spatial_shape)
        self.m_min = float(m_min)
        self.m_max = float(m_max)

        # Escala 1 mantém os canais da implementação original.
        encoder_channels = [
            max(1, round(self.in_channels * (2 ** (index + 1)) * encoder_channel_scale))
            for index in range(n_blocks_encoder)
        ]
        encoder_blocks = []
        previous_channels = self.in_channels
        for channels in encoder_channels:
            encoder_blocks.append(
                Block(
                    in_channels=previous_channels,
                    out_channels=channels,
                    stride=1,
                    operation="down",
                )
            )
            previous_channels = channels
        self.encoder = nn.ModuleList(encoder_blocks)
        latent_channels = encoder_channels[-1]

        # A escala do decodificador é aplicada após a divisão dos canais latentes.
        decoder_channels = [
            max(1, round((latent_channels // (2 ** (index + 1))) * decoder_channel_scale))
            for index in range(n_blocks_decoder)
        ]
        decoder_blocks = []
        previous_channels = latent_channels
        for index, channels in enumerate(decoder_channels):
            final_shape = (
                self.final_spatial_shape
                if index == len(decoder_channels) - 1
                else None
            )
            block = Block(
                in_channels=previous_channels,
                out_channels=channels,
                stride=1,
                operation="up",
                final_shape=final_shape,
            )
            block.layers[-1] = nn.Upsample(
                size=final_shape,
                scale_factor=None if final_shape is not None else 2,
                mode="bilinear",
                align_corners=False,
            )
            decoder_blocks.append(block)
            previous_channels = channels
        self.decoder = nn.ModuleList(decoder_blocks)

        # Caminho separado da atenção, com as mesmas larguras dos três
        # primeiros blocos principais e pesos próprios.
        attention_decoder_blocks = []
        previous_channels = latent_channels
        for channels in decoder_channels[:3]:
            block = Block(
                in_channels=previous_channels,
                out_channels=channels,
                stride=1,
                operation="up",
            )
            block.layers[-1] = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            attention_decoder_blocks.append(block)
            previous_channels = channels
        self.attention_decoder = nn.ModuleList(attention_decoder_blocks)

        # Cada operador combina o latente com uma saída do caminho separado.
        self.attention_branches = nn.ModuleList(
            _LatentGuidedAttention(
                latent_channels=latent_channels,
                feature_channels=channels,
                projection_channels=max(1, channels // attention_reduction),
                latent_kernel_size=attention_latent_kernel_size,
                feature_kernel_size=attention_feature_kernel_size,
                score_kernel_size=attention_score_kernel_size,
            )
            for channels in decoder_channels[:3]
        )

        fused_channels = sum(decoder_channels[:3]) + decoder_channels[-1]
        self.output_projection = nn.Conv2d(
            fused_channels,
            final_out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"A entrada do PGANN deve ter o formato [batch, {self.in_channels}, height, width]"
            )

        latent = x
        for block in self.encoder:
            latent = block(latent)

        attention_decoded = latent
        attention_features: list[torch.Tensor] = []
        for block in self.attention_decoder:
            attention_decoded = block(attention_decoded)
            attention_features.append(attention_decoded)

        decoded = latent
        for block in self.decoder:
            decoded = block(decoded)

        attended_features = [
            branch(latent, feature, self.final_spatial_shape)
            for branch, feature in zip(self.attention_branches, attention_features)
        ]
        fused = torch.cat([*attended_features, decoded], dim=1)
        normalized = torch.sigmoid(self.output_projection(fused))
        return self.m_min + normalized * (self.m_max - self.m_min)
