import pytest
import torch

from hs_tasnet.losses.metrics import compute_waveform_metrics


def test_metrics_contains_l1_and_sdr():
    pred = torch.randn(2, 4, 256)
    target = torch.randn(2, 4, 1, 256)
    metrics = compute_waveform_metrics(
        pred, target, metrics=("l1", "sdr"), target_channel_policy="strict"
    )
    assert set(metrics.keys()) == {"l1", "sdr"}
    assert torch.isfinite(metrics["l1"])
    assert torch.isfinite(metrics["sdr"])


def test_strict_policy_rejects_multichannel_target_for_mono_prediction():
    pred = torch.randn(1, 4, 64)
    target = torch.randn(1, 4, 2, 64)
    with pytest.raises(ValueError):
        compute_waveform_metrics(pred, target, metrics=("l1",), target_channel_policy="strict")


def test_mono_downmix_policy_accepts_multichannel_target():
    pred = torch.randn(1, 4, 64)
    target = torch.randn(1, 4, 2, 64)
    metrics = compute_waveform_metrics(
        pred, target, metrics=("l1",), target_channel_policy="mono_downmix"
    )
    assert "l1" in metrics
    assert torch.isfinite(metrics["l1"])
