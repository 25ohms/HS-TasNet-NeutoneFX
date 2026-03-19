import pytest
import torch

from hs_tasnet.infer.infer import _align_input_channels, _resolve_model_config


def test_resolve_model_config_prefers_checkpoint_model_config(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model": {},
            "cfg": {
                "model": {
                    "sample_rate": 44100,
                    "audio_channels": 1,
                    "num_stems": 4,
                    "stems": ["drums", "bass", "vocals", "other"],
                    "window_size": 1024,
                    "hop_size": 512,
                    "enc_channels": 1024,
                    "wave_lstm_hidden": 500,
                    "spec_lstm_hidden": 500,
                    "shared_lstm_hidden": 1000,
                    "post_split_wave_lstm_hidden": 500,
                    "post_split_spec_lstm_hidden": 500,
                    "wave_num_blocks": 1,
                    "spec_num_blocks": 1,
                    "shared_num_blocks": 1,
                    "post_split_num_blocks": 1,
                    "fusion": "concat",
                    "mask_activation": "sigmoid",
                    "spec_mask_representation": "magnitude",
                }
            },
        },
        checkpoint_path,
    )

    cfg = {"model": {"enc_channels": 64, "wave_lstm_hidden": 128}}

    model_cfg = _resolve_model_config(cfg, str(checkpoint_path), torch.device("cpu"))

    assert model_cfg.enc_channels == 1024
    assert model_cfg.wave_lstm_hidden == 500
    assert model_cfg.shared_lstm_hidden == 1000


def test_align_input_channels_downmixes_stereo_for_mono_model():
    audio = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    aligned = _align_input_channels(audio, expected_channels=1, channel_policy="mono_downmix")

    assert aligned.shape == (1, 1, 2)
    assert torch.allclose(aligned, torch.tensor([[[3.0, 5.0]]]))


def test_align_input_channels_rejects_stereo_in_strict_mode():
    audio = torch.randn(1, 2, 16)

    with pytest.raises(ValueError):
        _align_input_channels(audio, expected_channels=1, channel_policy="strict")
