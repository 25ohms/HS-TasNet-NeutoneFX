from __future__ import annotations

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.models.modules import ConvEncoder


def test_conv_encoder_exposes_frame_norms_alongside_features():
    encoder = ConvEncoder(in_channels=1, out_channels=16, kernel_size=128, stride=64)
    audio = torch.randn(2, 1, 1024)

    features, frame_norms = encoder.encode_with_norms(audio)

    assert features.shape == (2, 16, 15)
    assert frame_norms.shape == (2, 1, 15)
    assert torch.all(frame_norms > 0)
    assert torch.allclose(features, encoder(audio), atol=1e-6, rtol=1e-6)


def test_waveform_rescaling_restores_per_frame_energy_before_decoding():
    cfg = HSTasNetConfig(
        window_size=128,
        hop_size=64,
        enc_channels=16,
        wave_lstm_hidden=16,
        spec_lstm_hidden=16,
        shared_lstm_hidden=32,
        post_split_wave_lstm_hidden=16,
        post_split_spec_lstm_hidden=16,
    )
    model = HSTasNet(cfg)
    masked_conv = torch.ones(2, cfg.num_stems, cfg.enc_channels, 15)
    frame_norms = torch.full((2, 1, 15), 3.5)

    rescaled = model._apply_conv_frame_rescaling(masked_conv, frame_norms)

    assert rescaled.shape == masked_conv.shape
    assert torch.allclose(rescaled, masked_conv * 3.5)
