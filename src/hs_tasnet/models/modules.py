from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-8

        self.relu_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.gate_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.relu_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.shape[-1] < self.kernel_size:
            raise ValueError(
                f"Expected at least {self.kernel_size} samples for the waveform encoder, "
                f"got {audio.shape[-1]}"
            )

        # TasNet encodes normalized overlapping waveform segments rather than applying
        # a single unconstrained analysis convolution over the raw signal.
        segments = audio.unfold(dimension=-1, size=self.kernel_size, step=self.stride)
        norms = segments.pow(2).sum(dim=(1, 3), keepdim=True).sqrt().clamp_min(self.eps)
        segments = segments / norms

        bsz, channels, frames, window = segments.shape
        segments = segments.permute(0, 2, 1, 3).reshape(bsz, frames, channels * window)

        relu_weight = self.relu_weight.reshape(self.out_channels, channels * window)
        gate_weight = self.gate_weight.reshape(self.out_channels, channels * window)

        relu_branch = F.linear(segments, relu_weight)
        gate_branch = F.linear(segments, gate_weight)
        encoded = F.relu(relu_branch) * torch.sigmoid(gate_branch)
        return encoded.transpose(1, 2).contiguous()


class ConvDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.register_buffer("synthesis_window", torch.hann_window(kernel_size, periodic=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        window = self.synthesis_window.to(device=features.device, dtype=self.deconv.weight.dtype)
        weight = self.deconv.weight * window.view(1, 1, -1)
        return F.conv_transpose1d(
            features,
            weight,
            bias=self.deconv.bias,
            stride=self.deconv.stride,
            padding=self.deconv.padding,
            output_padding=self.deconv.output_padding,
            groups=self.deconv.groups,
            dilation=self.deconv.dilation,
        )


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


class BranchSplit(nn.Module):
    def __init__(self, fused_channels: int, conv_channels: int, spec_channels: int):
        super().__init__()
        self.conv_proj = nn.Conv1d(fused_channels, conv_channels, kernel_size=1)
        self.spec_proj = nn.Conv1d(fused_channels, spec_channels, kernel_size=1)

    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_proj(fused), self.spec_proj(fused)


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


class DomainMaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        num_stems: int,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.num_stems = num_stems
        self.head = nn.Conv1d(in_channels, num_stems * feature_channels, kernel_size=1)
        if activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = torch.sigmoid

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mask = self.head(features)
        return self.activation(mask)


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
