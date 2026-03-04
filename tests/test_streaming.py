import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.models.streaming import StreamingHSTasNet


def test_streaming_shapes():
    cfg = HSTasNetConfig(window_size=128, hop_size=64, enc_channels=16, lstm_hidden=32)
    model = HSTasNet(cfg)
    streamer = StreamingHSTasNet(model)

    hops = []
    for _ in range(4):
        hop = torch.randn(cfg.audio_channels, cfg.hop_size)
        out = streamer.step(hop)
        assert out.shape[-1] == cfg.hop_size
        hops.append(out)

    streamed = torch.cat(hops, dim=-1)
    assert streamed.shape[1] == cfg.num_stems
    assert streamed.shape[-1] == cfg.hop_size * 4
    assert torch.isfinite(streamed).all()
