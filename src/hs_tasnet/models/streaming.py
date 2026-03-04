from __future__ import annotations

from typing import List, Tuple

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
        self.buffer = torch.zeros(1, self.audio_channels, self.window_size)
        self.state: List[Tuple[torch.Tensor, torch.Tensor]] | None = None

    @torch.no_grad()
    def step(self, x_hop: torch.Tensor) -> torch.Tensor:
        if x_hop.dim() == 2:
            x_hop = x_hop.unsqueeze(0)
        if x_hop.shape[-1] != self.hop_size:
            raise ValueError(f"Expected hop size {self.hop_size}, got {x_hop.shape[-1]}")
        if x_hop.shape[1] != self.audio_channels:
            raise ValueError("Channel mismatch for streaming input")

        self.buffer = torch.cat([self.buffer[..., self.hop_size :], x_hop], dim=-1)
        y, aux = self.model(
            self.buffer,
            auto_curtail_length_to_multiple=False,
            return_aux=True,
            state=self.state,
        )
        self.state = aux.get("state")
        return y[..., -self.hop_size :]
