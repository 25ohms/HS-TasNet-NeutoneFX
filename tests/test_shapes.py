import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.models.modules import ConvDecoder, ConvEncoder


def test_default_encoder_width_matches_paper_phase_1():
    assert HSTasNetConfig().enc_channels == 1024


def test_gated_waveform_encoder_outputs_nonnegative_features():
    encoder = ConvEncoder(in_channels=1, out_channels=16, kernel_size=128, stride=64)
    x = torch.randn(2, 1, 1024)
    y = encoder(x)

    assert y.shape == (2, 16, 15)
    assert torch.all(y >= 0)


def test_waveform_decoder_uses_hann_windowed_synthesis_filters():
    decoder = ConvDecoder(in_channels=16, out_channels=1, kernel_size=128, stride=64)
    features = torch.randn(2, 16, 15)
    y = decoder(features)

    assert decoder.synthesis_window.shape == (128,)
    assert torch.isclose(decoder.synthesis_window[0], torch.tensor(0.0))
    assert torch.isclose(decoder.synthesis_window[-1], torch.tensor(0.0))
    assert y.shape == (2, 1, 1024)


def test_forward_shapes():
    cfg = HSTasNetConfig(window_size=128, hop_size=64, enc_channels=16, lstm_hidden=32)
    model = HSTasNet(cfg)
    x = torch.randn(2, cfg.audio_channels, 1024)
    y, aux = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == cfg.num_stems
    assert y.shape[2] == 1024
    assert aux["waveform_branch_features"].shape == (2, 16, 15)
    assert aux["spectral_branch_features"].shape == (2, 65, 15)
    assert aux["shared_features"].shape == (2, 81, 15)
    assert aux["split_conv_features"].shape == (2, 16, 15)
    assert aux["split_spec_features"].shape == (2, 65, 15)
