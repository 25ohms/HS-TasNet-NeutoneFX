from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from hs_tasnet.utils.audio import istft, stft


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio: [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.size(1) > 1:
            audio = audio.mean(dim=1, keepdim=True)
        spec = stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        magnitude = spec.abs()
        phase = torch.angle(spec)
        return magnitude, phase


class SpectrogramDecoder(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        complex_spec = torch.polar(magnitude, phase)
        audio = istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=length,
        )
        return audio


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return self.conv(audio)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.deconv(features)


class MemoryLSTMBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, channels)

    def forward(
        self, features: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # features: [B, C, T]
        x = features.transpose(1, 2)  # [B, T, C]
        out, new_state = self.lstm(x, state)
        out = self.proj(out).transpose(1, 2)
        return out + features, new_state


class Fusion(nn.Module):
    def __init__(self, mode: str, conv_channels: int, spec_channels: int):
        super().__init__()
        self.mode = mode
        if mode == "sum":
            self.spec_proj = nn.Conv1d(spec_channels, conv_channels, kernel_size=1)
        else:
            self.spec_proj = None

    def forward(self, conv_features: torch.Tensor, spec_features: torch.Tensor) -> torch.Tensor:
        # Align time dimension
        if spec_features.shape[-1] != conv_features.shape[-1]:
            spec_features = torch.nn.functional.interpolate(
                spec_features, size=conv_features.shape[-1], mode="linear", align_corners=False
            )
        if self.mode == "sum":
            spec_features = self.spec_proj(spec_features)
            return conv_features + spec_features
        return torch.cat([conv_features, spec_features], dim=1)


class MaskHead(nn.Module):
    def __init__(
        self,
        fused_channels: int,
        conv_channels: int,
        spec_channels: int,
        num_stems: int,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.num_stems = num_stems
        self.conv_head = nn.Conv1d(fused_channels, num_stems * conv_channels, kernel_size=1)
        self.spec_head = nn.Conv1d(fused_channels, num_stems * spec_channels, kernel_size=1)
        if activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = torch.sigmoid

    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_mask = self.conv_head(fused)
        spec_mask = self.spec_head(fused)
        conv_mask = self.activation(conv_mask)
        spec_mask = self.activation(spec_mask)
        return conv_mask, spec_mask


class HybridCombiner(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, conv_audio: torch.Tensor, spec_audio: torch.Tensor) -> torch.Tensor:
        # conv_audio/spec_audio: [B, S, T]
        if conv_audio.shape[-1] != spec_audio.shape[-1]:
            min_len = min(conv_audio.shape[-1], spec_audio.shape[-1])
            conv_audio = conv_audio[..., :min_len]
            spec_audio = spec_audio[..., :min_len]
        return self.alpha * conv_audio + (1 - self.alpha) * spec_audio
