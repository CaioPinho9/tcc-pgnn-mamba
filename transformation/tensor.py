import torch


def as_float_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)

    return torch.as_tensor(data, dtype=torch.float32)