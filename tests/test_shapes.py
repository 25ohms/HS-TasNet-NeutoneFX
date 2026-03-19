import pytest
import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.models.modules import ConvDecoder, ConvEncoder, HybridCombiner, MemoryLSTMBlock


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


def test_memory_block_uses_two_lstm_stages():
    block = MemoryLSTMBlock(channels=16, hidden_size=32, skip_mode="identity")
    x = torch.randn(2, 16, 15)
    y, state = block(x)

    assert y.shape == x.shape
    assert isinstance(state, tuple)
    assert len(state) == 2
    assert state[0][0].shape[-1] == 32
    assert state[1][0].shape[-1] == 32


def test_memory_block_encoded_skip_accepts_encoded_representation():
    block = MemoryLSTMBlock(channels=16, hidden_size=32, skip_mode="encoded", skip_channels=16)
    x = torch.randn(2, 16, 15)
    encoded = torch.randn(2, 16, 15)
    y, _ = block(x, encoded_representation=encoded)
    assert y.shape == x.shape


def test_memory_block_encoded_skip_requires_encoded_representation():
    block = MemoryLSTMBlock(channels=16, hidden_size=32, skip_mode="encoded", skip_channels=16)
    x = torch.randn(2, 16, 15)
    with pytest.raises(ValueError):
        block(x)


def test_hybrid_combiner_uses_direct_sum():
    combiner = HybridCombiner()
    conv_audio = torch.ones(2, 4, 16)
    spec_audio = 2 * torch.ones(2, 4, 16)
    y = combiner(conv_audio, spec_audio)
    assert torch.allclose(y, 3 * torch.ones_like(y))


def test_forward_shapes():
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
    x = torch.randn(2, cfg.audio_channels, 1024)
    y = model(x)
    output = model.separate(x)
    assert y.shape[0] == 2
    assert y.shape[1] == cfg.num_stems
    assert y.shape[2] == 1024
    assert output.waveform_branch_features.shape == (2, 16, 15)
    assert output.spectral_branch_features.shape == (2, 65, 15)
    assert output.shared_features.shape == (2, 81, 15)
    assert output.split_conv_features.shape == (2, 16, 15)
    assert output.split_spec_features.shape == (2, 65, 15)
