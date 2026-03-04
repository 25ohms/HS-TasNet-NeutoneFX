import torch


def random_gain(x: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2) -> torch.Tensor:
    gain = torch.empty(1).uniform_(min_gain, max_gain).item()
    return x * gain
