from __future__ import annotations

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet


class StreamingHSTasNet:
    def __init__(self, model: HSTasNet):
        self.model = model
        self.hop_size = model.cfg.hop_size
        self.window_size = model.cfg.window_size
        self.audio_channels = model.cfg.audio_channels
        self.reset()

    def reset(self) -> None:
        self.stream_state = self.model.init_stream_state()
        self.state = None

    @torch.no_grad()
    def step(self, x_hop: torch.Tensor) -> torch.Tensor:
        if x_hop.dim() == 2:
            x_hop = x_hop.unsqueeze(0)
        if x_hop.shape[-1] != self.hop_size:
            raise ValueError(f"Expected hop size {self.hop_size}, got {x_hop.shape[-1]}")
        if x_hop.shape[1] != self.audio_channels:
            raise ValueError("Channel mismatch for streaming input")

        y_hop, self.stream_state = self.model.stream_step(x_hop, self.stream_state)
        self.state = self.stream_state.get("branch_state")
        return y_hop
