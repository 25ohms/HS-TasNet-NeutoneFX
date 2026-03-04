import torch


def resolve_device(name: str | None = None) -> torch.device:
    if name is None:
        name = "cuda"
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
