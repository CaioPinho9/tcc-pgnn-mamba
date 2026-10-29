import torch

from transformation.normalizer import Normalizer
from transformation.tensor import as_float_tensor


class MinMaxNormalizer(Normalizer):
    def fit(self, data):
        data = as_float_tensor(data)
        self.data_min = data.min()
        self.data_max = data.max()
        return self

    def fit_normalize(self, data):
        return self.fit(data).normalize(data)

    def normalize(self, data):
        data = as_float_tensor(data)
        if self.data_min is None or self.data_max is None:
            raise ValueError("O normalizador ainda não foi ajustado. Chame fit() primeiro.")

        d_min = self.data_min.to(device=data.device)
        d_max = self.data_max.to(device=data.device)

        data_01 = (data - d_min) / (d_max - d_min + self.eps)
        return self.target_min + data_01 * (self.target_max - self.target_min)

    def denormalize(self, normalized_data):
        normalized_data = as_float_tensor(normalized_data)
        if self.data_min is None or self.data_max is None:
            raise ValueError("O normalizador ainda não foi ajustado. Não é possível desnormalizar.")

        d_min = self.data_min.to(device=normalized_data.device)
        d_max = self.data_max.to(device=normalized_data.device)

        data_01 = (normalized_data - self.target_min) / (self.target_max - self.target_min + self.eps)
        return d_min + data_01 * (d_max - d_min)

    def get_state(self) -> dict:
        return {
            "data_min": float(self.data_min.detach().cpu()) if self.data_min is not None else None,
            "data_max": float(self.data_max.detach().cpu()) if self.data_max is not None else None,
            "target_min": float(self.target_min),
            "target_max": float(self.target_max),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state(cls, state: dict, device: str | torch.device = "cpu"):
        normalizer = cls(
            target_min=state.get("target_min", -1.0),
            target_max=state.get("target_max", 1.0),
            eps=state.get("eps", 1e-8),
        )
        if state.get("data_min") is not None:
            normalizer.data_min = torch.tensor(float(state["data_min"]), device=device)
        if state.get("data_max") is not None:
            normalizer.data_max = torch.tensor(float(state["data_max"]), device=device)
        return normalizer
