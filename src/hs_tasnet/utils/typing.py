from typing import Dict, Tuple

import torch

TensorDict = Dict[str, torch.Tensor]
State = Tuple[torch.Tensor, torch.Tensor]
