from __future__ import annotations

import time
from typing import Dict

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet
from hs_tasnet.models.streaming import StreamingHSTasNet


def build_streaming(model: HSTasNet) -> StreamingHSTasNet:
    model.eval()
    return StreamingHSTasNet(model)


@torch.no_grad()
def stream_step(streamer: StreamingHSTasNet, hop_audio: torch.Tensor) -> torch.Tensor:
    return streamer.step(hop_audio)


@torch.no_grad()
def benchmark_streaming(
    streamer: StreamingHSTasNet,
    num_hops: int = 200,
    warmup_hops: int = 20,
    synchronize_cuda: bool = True,
) -> Dict[str, float]:
    hop_size = streamer.hop_size
    sample_rate = streamer.model.sample_rate
    device = next(streamer.model.parameters()).device

    def _maybe_sync() -> None:
        if synchronize_cuda and device.type == "cuda":
            torch.cuda.synchronize(device=device)

    for _ in range(max(warmup_hops, 0)):
        hop = torch.randn(1, streamer.audio_channels, hop_size, device=device)
        streamer.step(hop)
    _maybe_sync()

    start = time.perf_counter()
    for _ in range(max(num_hops, 1)):
        hop = torch.randn(1, streamer.audio_channels, hop_size, device=device)
        streamer.step(hop)
    _maybe_sync()
    elapsed = time.perf_counter() - start

    avg_hop_ms = (elapsed / max(num_hops, 1)) * 1000.0
    hop_duration_s = hop_size / float(sample_rate)
    rtf = hop_duration_s / max(elapsed / max(num_hops, 1), 1e-12)
    return {
        "avg_hop_ms": avg_hop_ms,
        "rtf": rtf,
        "algorithmic_latency_ms": (streamer.window_size / float(sample_rate)) * 1000.0,
    }
