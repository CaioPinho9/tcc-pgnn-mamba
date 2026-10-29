from .pinn_fwi.networks import (
    Autoencoder as PgnnMardan2024Autoencoder,
)
from .pgann_mardan_2024 import PGANNMardan2024
from .unet import UNet

MAMBA_MODEL_NAMES = {"unetmamba", "mamba"}
PGNN_MARDAN_2024_MODEL_NAME = "pgnn_mardan_2024"
PGANN_MARDAN_2024_MODEL_NAME = "pgann_mardan_2024"
UNET_MODEL_NAME = "unet"
MARDAN_2024_MODEL_NAMES = frozenset({
    PGNN_MARDAN_2024_MODEL_NAME,
    PGANN_MARDAN_2024_MODEL_NAME,
})


def is_mamba_model(model_name: str) -> bool:
    return model_name in MAMBA_MODEL_NAMES


def _build_pgnn_mardan_2024(hyperparameters: dict, device):
    return PgnnMardan2024Autoencoder(
        in_channels=hyperparameters["in_channels"],
        n_blocks_encoder=hyperparameters["n_blocks_encoder"],
        n_blocks_decoder=hyperparameters["n_blocks_decoder"],
        final_size_encoder=hyperparameters["final_size_encoder"],
        initial_shape_decoder=hyperparameters["decoder_initial_shape"],
        nt=hyperparameters["nt"],
        nr=hyperparameters["nr"],
        final_spatial_shape=hyperparameters["final_spatial_shape"],
        m_min=hyperparameters.get(
            "model_output_min",
            hyperparameters.get("m_min", 0.0),
        ),
        m_max=hyperparameters.get(
            "model_output_max",
            hyperparameters.get("m_max", 1.0),
        ),
        final_out_channels=hyperparameters["final_out_channels"],
        batch_size=hyperparameters.get(
            "model_batch_size",
            hyperparameters.get("batch_size", 1),
        ),
    ).to(device)


def build_pgnn_mardan_2024(
        hyperparameters: dict,
        device,
) -> PgnnMardan2024Autoencoder:
    return _build_pgnn_mardan_2024(hyperparameters, device)


def build_pgann_mardan_2024(
        hyperparameters: dict,
        device,
) -> PGANNMardan2024:
    return PGANNMardan2024(
        in_channels=hyperparameters["in_channels"],
        n_blocks_encoder=hyperparameters["n_blocks_encoder"],
        n_blocks_decoder=hyperparameters["n_blocks_decoder"],
        final_spatial_shape=hyperparameters["final_spatial_shape"],
        m_min=hyperparameters["model_output_min"],
        m_max=hyperparameters["model_output_max"],
        final_out_channels=hyperparameters["final_out_channels"],
        encoder_channel_scale=hyperparameters["pgann_encoder_channel_scale"],
        decoder_channel_scale=hyperparameters["pgann_decoder_channel_scale"],
        attention_latent_kernel_size=hyperparameters[
            "pgann_attention_latent_kernel_size"
        ],
        attention_feature_kernel_size=hyperparameters[
            "pgann_attention_feature_kernel_size"
        ],
        attention_score_kernel_size=hyperparameters[
            "pgann_attention_score_kernel_size"
        ],
        attention_reduction=hyperparameters["pgann_attention_reduction"],
    ).to(device)


def build_unet(hyperparameters: dict, device) -> UNet:
    return UNet(
        in_channels=hyperparameters["in_channels"],
        n_blocks_encoder=hyperparameters["n_blocks_encoder"],
        n_blocks_decoder=hyperparameters["n_blocks_decoder"],
        final_spatial_shape=hyperparameters["final_spatial_shape"],
        final_out_channels=hyperparameters["final_out_channels"],
    ).to(device)


def _build_mamba_model(hyperparameters: dict, device):
    from .mamba import MambaUNet

    return MambaUNet(
        in_channels=hyperparameters["in_channels"],
        n_blocks_encoder=hyperparameters["n_blocks_encoder"],
        n_blocks_decoder=hyperparameters["n_blocks_decoder"],
        final_spatial_shape=hyperparameters["final_spatial_shape"],
        base_channels=hyperparameters.get(
            "mamba_base_channels",
            hyperparameters.get("base_channels", 16),
        ),
        final_out_channels=hyperparameters["final_out_channels"],
        d_state=hyperparameters.get("mamba_d_state", 16),
        use_mamba_encoder=hyperparameters["use_mamba_encoder"],
        use_mamba_bottleneck=hyperparameters["use_mamba_bottleneck"],
        use_mamba_decoder=hyperparameters["use_mamba_decoder"],
        use_skip_connections=hyperparameters.get("use_skip_connections", False),
        d_conv=hyperparameters["mamba_d_conv"],
        expand=hyperparameters["mamba_expand"],
        dropout=hyperparameters["mamba_dropout"],
        use_pos_embed=hyperparameters["use_pos_embed"],
        conv_kernel_size=hyperparameters.get("mamba_conv_kernel_size", 3),
    ).to(device)


def build_mamba(hyperparameters: dict, device):
    return _build_mamba_model(hyperparameters, device)


def build_model(model_name: str, hyperparameters: dict, device):
    if model_name == PGNN_MARDAN_2024_MODEL_NAME:
        return build_pgnn_mardan_2024(hyperparameters, device)
    if model_name == PGANN_MARDAN_2024_MODEL_NAME:
        return build_pgann_mardan_2024(hyperparameters, device)
    if model_name == UNET_MODEL_NAME:
        return build_unet(hyperparameters, device)
    if model_name in MAMBA_MODEL_NAMES:
        return build_mamba(hyperparameters, device)
    raise ValueError(f"Nome de modelo desconhecido (model_name): {model_name!r}")
