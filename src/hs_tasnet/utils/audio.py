from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str | pathlib.Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    data = data.T  # [C, T]
    if target_sr is not None and sr != target_sr:
        raise ValueError(f"Expected sample rate {target_sr}, got {sr}.")
    return data.astype(np.float32), sr


def save_audio(path: str | pathlib.Path, audio: np.ndarray, sample_rate: int) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.T  # [T, C]
    sf.write(str(path), audio, sample_rate)


def stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
) -> torch.Tensor:
    window = torch.hann_window(win_length or n_fft, device=audio.device)
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        center=False,
        return_complex=True,
    )


def istft(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    length: int | None = None,
) -> torch.Tensor:
    window = torch.hann_window(win_length or n_fft, device=spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        center=False,
        length=length,
    )


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    length = x.shape[-1]
    pad = (-length) % multiple
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad))
