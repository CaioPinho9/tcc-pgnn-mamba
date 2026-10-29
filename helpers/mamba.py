from __future__ import annotations

import importlib.machinery
import importlib.util
import sys


def load_mamba():
    """Carrega a camada Mamba padrão do pacote mamba-ssm instalado."""
    package = sys.modules.get("mamba_ssm")
    if package is None:
        spec = importlib.machinery.PathFinder.find_spec("mamba_ssm", sys.path)
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError("mamba-ssm é necessário para o modelo Mamba")
        package = importlib.util.module_from_spec(spec)
        sys.modules["mamba_ssm"] = package

    from mamba_ssm.modules.mamba_simple import Mamba

    return Mamba
