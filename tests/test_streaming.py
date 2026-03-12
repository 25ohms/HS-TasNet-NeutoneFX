import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.models.streaming import StreamingHSTasNet


def test_streaming_shapes():
    cfg = HSTasNetConfig(
        window_size=128,
        hop_size=64,
        enc_channels=16,
        wave_lstm_hidden=32,
        spec_lstm_hidden=32,
        shared_lstm_hidden=64,
        post_split_wave_lstm_hidden=32,
        post_split_spec_lstm_hidden=32,
    )
    model = HSTasNet(cfg)
    streamer = StreamingHSTasNet(model)

    hops = []
    for _ in range(4):
        hop = torch.randn(cfg.audio_channels, cfg.hop_size)
        out = streamer.step(hop)
        assert out.shape[-1] == cfg.hop_size
        hops.append(out)

    streamed = torch.cat(hops, dim=-1)
    assert set(streamer.state.keys()) == {
        "waveform_branch",
        "spectral_branch",
        "shared_branch",
        "post_split_wave_branch",
        "post_split_spec_branch",
    }
    assert streamed.shape[1] == cfg.num_stems
    assert streamed.shape[-1] == cfg.hop_size * 4
    assert torch.isfinite(streamed).all()


def test_streaming_matches_offline_with_streaming_padding():
    torch.manual_seed(0)
    cfg = HSTasNetConfig(
        window_size=128,
        hop_size=64,
        enc_channels=16,
        wave_lstm_hidden=32,
        spec_lstm_hidden=32,
        shared_lstm_hidden=64,
        post_split_wave_lstm_hidden=32,
        post_split_spec_lstm_hidden=32,
    )
    model = HSTasNet(cfg)
    model.eval()
    streamer = StreamingHSTasNet(model)

    num_hops = 6
    signal = torch.randn(1, cfg.audio_channels, cfg.hop_size * num_hops)
    streamed_hops = []
    for hop_idx in range(num_hops):
        hop = signal[..., hop_idx * cfg.hop_size : (hop_idx + 1) * cfg.hop_size]
        streamed_hops.append(streamer.step(hop))
    streamed = torch.cat(streamed_hops, dim=-1)

    overlap = cfg.window_size - cfg.hop_size
    leading_zeros = torch.zeros(1, cfg.audio_channels, overlap)
    offline_input = torch.cat([leading_zeros, signal], dim=-1)
    offline = model(offline_input, auto_curtail_length_to_multiple=False)[..., : signal.shape[-1]]

    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-4, rtol=1e-4)
