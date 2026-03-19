from __future__ import annotations

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.optim import build_optimizer, get_optim_config
from hs_tasnet.train.regularization import compute_singular_value_penalty


def test_optim_config_uses_demucs_style_block_and_backfills_train_fields():
    cfg = {
        "train": {"lr": 1e-4, "grad_clip_norm": 1.0},
        "optim": {"optim": "adam", "loss": "l1"},
    }
    optim_cfg = get_optim_config(cfg)

    assert optim_cfg["lr"] == 1e-4
    assert optim_cfg["clip_grad"] == 1.0
    assert optim_cfg["beta1"] == 0.9
    assert optim_cfg["beta2"] == 0.999


def test_build_optimizer_supports_adamw():
    model = torch.nn.Linear(8, 4)
    cfg = {"optim": {"optim": "adamw", "lr": 3e-4, "weight_decay": 0.01}}

    optimizer = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_singular_value_regularization_is_zero_when_disabled():
    model = torch.nn.Linear(8, 4)
    cfg = {"regularization": {"singular_value": {"enabled": False}}}

    penalty = compute_singular_value_penalty(model, cfg)

    assert penalty.shape == ()
    assert penalty.item() == 0.0


def test_singular_value_regularization_targets_matching_weights():
    model = HSTasNet(
        HSTasNetConfig(
            window_size=128,
            hop_size=64,
            enc_channels=16,
            wave_lstm_hidden=16,
            spec_lstm_hidden=16,
            shared_lstm_hidden=32,
            post_split_wave_lstm_hidden=16,
            post_split_spec_lstm_hidden=16,
        )
    )
    cfg = {
        "regularization": {
            "singular_value": {
                "enabled": True,
                "weight": 1e-4,
                "interval": 1,
                "rank": 1,
                "niter": 1,
                "target_patterns": ["conv_encoder"],
                "max_entries": 4,
            }
        }
    }

    penalty = compute_singular_value_penalty(model, cfg)

    assert penalty.shape == ()
    assert penalty.item() >= 0.0


def test_bottleneck_group_norm_is_constructed_when_enabled():
    model = HSTasNet(
        HSTasNetConfig(
            window_size=128,
            hop_size=64,
            enc_channels=16,
            wave_lstm_hidden=16,
            spec_lstm_hidden=16,
            shared_lstm_hidden=32,
            post_split_wave_lstm_hidden=16,
            post_split_spec_lstm_hidden=16,
            bottleneck_group_norm_groups=4,
        )
    )

    assert isinstance(model.shared_norm, torch.nn.GroupNorm)
    assert isinstance(model.split_conv_norm, torch.nn.GroupNorm)
