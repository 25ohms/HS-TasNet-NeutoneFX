import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig


def test_forward_shapes():
    cfg = HSTasNetConfig(window_size=128, hop_size=64, enc_channels=16, lstm_hidden=32)
    model = HSTasNet(cfg)
    x = torch.randn(2, cfg.audio_channels, 1024)
    y, _ = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == cfg.num_stems
    assert y.shape[2] == 1024
