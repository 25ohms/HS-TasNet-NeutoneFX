from __future__ import annotations

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet
from hs_tasnet.models.streaming import StreamingHSTasNet


def build_streaming(model: HSTasNet) -> StreamingHSTasNet:
    model.eval()
    return StreamingHSTasNet(model)


@torch.no_grad()
def stream_step(streamer: StreamingHSTasNet, hop_audio: torch.Tensor) -> torch.Tensor:
    return streamer.step(hop_audio)
