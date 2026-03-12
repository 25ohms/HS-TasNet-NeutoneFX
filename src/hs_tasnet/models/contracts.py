from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

LSTMState = Tuple[torch.Tensor, torch.Tensor]
MemoryBlockState = Tuple[LSTMState, LSTMState]
BranchState = Dict[str, List[MemoryBlockState] | None]


@dataclass
class HSTasNetOutput:
    audio: torch.Tensor
    conv_mask: torch.Tensor
    spec_mask: torch.Tensor
    state: BranchState
    waveform_branch_features: torch.Tensor
    spectral_branch_features: torch.Tensor
    shared_features: torch.Tensor
    split_conv_features: torch.Tensor
    split_spec_features: torch.Tensor

    def as_aux_dict(self) -> Dict[str, object]:
        return {
            "conv_mask": self.conv_mask,
            "spec_mask": self.spec_mask,
            "state": self.state,
            "waveform_branch_features": self.waveform_branch_features,
            "spectral_branch_features": self.spectral_branch_features,
            "shared_features": self.shared_features,
            "split_conv_features": self.split_conv_features,
            "split_spec_features": self.split_spec_features,
        }
