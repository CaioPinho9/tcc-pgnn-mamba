from abc import ABC, abstractmethod


class Normalizer(ABC):
    def __init__(self, target_min=-1.0, target_max=1.0, eps=1e-8):
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps
        self.data_min = None
        self.data_max = None

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def fit_normalize(self, data):
        pass

    @abstractmethod
    def normalize(self, data):
        pass

    @abstractmethod
    def denormalize(self, normalized_data):
        pass

    @abstractmethod
    def get_state(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict):
        pass