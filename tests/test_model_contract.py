import pytest
import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig


def test_forward_returns_audio_tensor_only():
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
    x = torch.randn(2, cfg.audio_channels, 1024)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, cfg.num_stems, 1024)


def test_separate_returns_structured_output_with_state():
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
    x = torch.randn(2, cfg.audio_channels, 1024)
    output = model.separate(x)

    assert output.audio.shape == (2, cfg.num_stems, 1024)
    assert set(output.state.keys()) == {
        "waveform_branch",
        "spectral_branch",
        "shared_branch",
        "post_split_wave_branch",
        "post_split_spec_branch",
    }
    assert output.conv_mask.shape[1] == cfg.num_stems
    assert output.spec_mask.shape[1] == cfg.num_stems


def test_stream_step_contract():
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
    stream_state = model.init_stream_state()
    hop = torch.randn(1, cfg.audio_channels, cfg.hop_size)

    y_hop, stream_state = model.stream_step(hop, stream_state)
    assert y_hop.shape == (1, cfg.num_stems, cfg.hop_size)
    assert "buffer" in stream_state
    assert "branch_state" in stream_state
    assert "conv_overlap" in stream_state
    assert "spec_overlap" in stream_state


def test_invalid_fusion_mode_rejected():
    with pytest.raises(ValueError):
        HSTasNet(HSTasNetConfig(fusion="invalid"))


def test_invalid_spec_mask_representation_rejected():
    with pytest.raises(ValueError):
        HSTasNet(HSTasNetConfig(spec_mask_representation="complex"))


def test_stem_count_must_match_num_stems():
    with pytest.raises(ValueError):
        HSTasNet(HSTasNetConfig(num_stems=4, stems=["drums", "bass"]))


def test_channel_mismatch_rejected_in_forward():
    cfg = HSTasNetConfig(audio_channels=1, window_size=128, hop_size=64, enc_channels=16)
    model = HSTasNet(cfg)
    x = torch.randn(1, 2, 512)
    with pytest.raises(ValueError):
        model(x)
