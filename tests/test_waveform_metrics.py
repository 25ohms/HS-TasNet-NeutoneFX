from __future__ import annotations

import torch

from hs_tasnet.losses.waveform import signal_distortion_ratio


def test_signal_distortion_ratio_prefers_better_predictions():
    target = torch.tensor([[[1.0, -1.0, 0.5, -0.5]]], dtype=torch.float32)
    good_pred = target.clone()
    bad_pred = torch.zeros_like(target)

    good_sdr = signal_distortion_ratio(good_pred, target)
    bad_sdr = signal_distortion_ratio(bad_pred, target)

    assert torch.isfinite(good_sdr)
    assert torch.isfinite(bad_sdr)
    assert good_sdr > bad_sdr
