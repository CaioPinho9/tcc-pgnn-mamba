from transformation.normalizer import Normalizer
from transformation.tensor import as_float_tensor


class AmplitudeNormalizer(Normalizer):
    def __init__(self):
        super().__init__()
        self.scale = None

    def fit(self, data):
        data = as_float_tensor(data)
        self.scale = data.abs().max()
        return self

    def fit_normalize(self, data):
        return self.fit(data).normalize(data)

    def normalize(self, data):
        data = as_float_tensor(data)
        if self.scale is None:
            raise ValueError("O normalizador ainda não foi ajustado. Chame fit() primeiro.")

        sc = self.scale.to(device=data.device)
        return data / sc.clamp_min(self.eps)

    def denormalize(self, normalized_data):
        normalized_data = as_float_tensor(normalized_data)
        if self.scale is None:
            raise ValueError("O normalizador ainda não foi ajustado. Não é possível desnormalizar.")

        sc = self.scale.to(device=normalized_data.device)
        return normalized_data * sc
