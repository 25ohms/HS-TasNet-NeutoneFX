from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from hs_tasnet.losses.waveform import l1_loss, signal_distortion_ratio


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_sdr = 0.0
    count = 0
    with torch.no_grad():
        for mixture, stems in loader:
            mixture = mixture.to(device)
            stems = stems.to(device)
            pred, _ = model(mixture)
            loss = l1_loss(pred, stems)
            sdr = signal_distortion_ratio(pred, stems)
            total_loss += loss.item()
            total_sdr += sdr.item()
            count += 1
    model.train()
    return {
        "l1": total_loss / max(count, 1),
        "sdr": total_sdr / max(count, 1),
    }
