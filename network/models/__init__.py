from .factory import (
    MAMBA_MODEL_NAMES,
    MARDAN_2024_MODEL_NAMES,
    PGANN_MARDAN_2024_MODEL_NAME,
    PGNN_DHARA_2022_MODEL_NAME,
    build_mamba,
    build_model,
    build_pgann_mardan_2024,
    build_pgnn_dhara_2022,
    build_unet,
    is_mamba_model,
)
from .pinn_fwi.networks import (
    Autoencoder as PgnnDhara2022Autoencoder,
)
from .pgann_mardan_2024 import PGANNMardan2024
from .unet import UNet


def __getattr__(name: str):
    if name == "MambaUNet":
        from .mamba import MambaUNet

        return MambaUNet
    raise AttributeError(f"o módulo {__name__!r} não possui o atributo {name!r}")

__all__ = [
    "MAMBA_MODEL_NAMES",
    "MARDAN_2024_MODEL_NAMES",
    "PGANN_MARDAN_2024_MODEL_NAME",
    "PGNN_DHARA_2022_MODEL_NAME",
    "build_mamba",
    "PgnnDhara2022Autoencoder",
    "PGANNMardan2024",
    "MambaUNet",
    "UNet",
    "build_model",
    "build_pgann_mardan_2024",
    "build_pgnn_dhara_2022",
    "build_unet",
    "is_mamba_model",
]
