from __future__ import annotations

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.train import _build_loader_kwargs, _resolve_amp_enabled
from hs_tasnet.utils.config import load_config


def test_train_paper_uses_only_optimization_loss_for_train_metrics():
    cfg = load_config("src/hs_tasnet/config/train_paper.yaml")

    assert cfg["train"]["metrics"] == ["l1"]
    assert cfg["eval"]["metrics"] == ["l1", "sdr"]


def test_amp_defaults_to_cuda_only_when_unspecified():
    cfg = {"device": {}}

    assert _resolve_amp_enabled(cfg, torch.device("cuda")) is True
    assert _resolve_amp_enabled(cfg, torch.device("cpu")) is False


def test_amp_respects_explicit_override():
    cfg = {"device": {"use_amp": False}}

    assert _resolve_amp_enabled(cfg, torch.device("cuda")) is False


def test_build_loader_kwargs_enables_persistent_workers_when_possible():
    cfg = {
        "train": {"batch_size": 4},
        "data": {
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        },
    }

    loader_kwargs = _build_loader_kwargs(cfg)

    assert loader_kwargs["batch_size"] == 4
    assert loader_kwargs["num_workers"] == 2
    assert loader_kwargs["prefetch_factor"] == 2
    assert loader_kwargs["persistent_workers"] is True


def test_build_loader_kwargs_skips_worker_only_options_for_single_process_loading():
    cfg = {
        "train": {"batch_size": 4},
        "data": {
            "num_workers": 0,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        },
    }

    loader_kwargs = _build_loader_kwargs(cfg)

    assert "prefetch_factor" not in loader_kwargs
    assert "persistent_workers" not in loader_kwargs


def test_batched_conv_decoder_matches_per_stem_reference():
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
    masked_conv = torch.randn(2, cfg.num_stems, cfg.enc_channels, 15)

    batched = model._decode_conv_batched(masked_conv)
    reference = torch.stack(
        [
            model.conv_decoder(masked_conv[:, stem_idx])
            for stem_idx in range(cfg.num_stems)
        ],
        dim=1,
    ).squeeze(2)

    assert batched.shape == reference.shape
    assert torch.allclose(batched, reference, atol=1e-5, rtol=1e-5)


def test_batched_spec_decoder_matches_per_stem_reference():
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
    batch_size = 2
    time_steps = 15
    spec_bins = cfg.window_size // 2 + 1
    masked_spec = torch.rand(batch_size, cfg.num_stems, spec_bins, time_steps)
    phase = torch.randn(batch_size, spec_bins, time_steps)
    audio_length = cfg.hop_size * time_steps

    batched = model._decode_spec_batched(masked_spec, phase, audio_length)
    reference = torch.stack(
        [
            model.spec_decoder(masked_spec[:, stem_idx], phase, length=audio_length)
            for stem_idx in range(cfg.num_stems)
        ],
        dim=1,
    )

    assert batched.shape == reference.shape
    assert torch.allclose(batched, reference, atol=1e-5, rtol=1e-5)
