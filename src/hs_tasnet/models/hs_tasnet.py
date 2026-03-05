from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from hs_tasnet.models.modules import (
    ConvDecoder,
    ConvEncoder,
    Fusion,
    HybridCombiner,
    MaskHead,
    MemoryLSTMBlock,
    SpectrogramDecoder,
    SpectrogramEncoder,
)
from hs_tasnet.utils.audio import pad_to_multiple


@dataclass
class HSTasNetConfig:
    sample_rate: int = 44100
    audio_channels: int = 1
    num_stems: int = 4
    stems: List[str] | None = None
    window_size: int = 1024
    hop_size: int = 512
    enc_channels: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    fusion: str = "concat"
    mask_activation: str = "sigmoid"


class HSTasNet(nn.Module):
    def __init__(self, cfg: HSTasNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or HSTasNetConfig()
        self.sample_rate = self.cfg.sample_rate
        self.audio_channels = self.cfg.audio_channels
        self.num_sources = self.cfg.num_stems
        self.stems = self.cfg.stems or ["drums", "bass", "vocals", "other"]
        self.segment_len = self.cfg.hop_size

        self.spec_encoder = SpectrogramEncoder(self.cfg.window_size, self.cfg.hop_size)
        self.spec_decoder = SpectrogramDecoder(self.cfg.window_size, self.cfg.hop_size)

        self.conv_encoder = ConvEncoder(
            in_channels=self.cfg.audio_channels,
            out_channels=self.cfg.enc_channels,
            kernel_size=self.cfg.window_size,
            stride=self.cfg.hop_size,
        )
        self.conv_decoder = ConvDecoder(
            in_channels=self.cfg.enc_channels,
            out_channels=self.cfg.audio_channels,
            kernel_size=self.cfg.window_size,
            stride=self.cfg.hop_size,
        )

        spec_channels = self.cfg.window_size // 2 + 1
        fused_channels = (
            self.cfg.enc_channels + spec_channels
            if self.cfg.fusion == "concat"
            else self.cfg.enc_channels
        )
        self.fusion = Fusion(self.cfg.fusion, self.cfg.enc_channels, spec_channels)
        self.memory_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(fused_channels, self.cfg.lstm_hidden)
                for _ in range(self.cfg.lstm_layers)
            ]
        )
        self.mask_head = MaskHead(
            fused_channels,
            conv_channels=self.cfg.enc_channels,
            spec_channels=spec_channels,
            num_stems=self.cfg.num_stems,
            activation=self.cfg.mask_activation,
        )
        self.combiner = HybridCombiner(alpha=0.5)

    def forward(
        self,
        audio: torch.Tensor,
        auto_curtail_length_to_multiple: bool = True,
        return_aux: bool = True,
        state: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        # audio: [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        orig_len = audio.shape[-1]
        if auto_curtail_length_to_multiple:
            audio = pad_to_multiple(audio, self.cfg.hop_size)

        conv_features = self.conv_encoder(audio)  # [B, Cc, Tenc]
        spec_mag, spec_phase = self.spec_encoder(audio)  # [B, F, Tspec]
        spec_features = spec_mag
        if spec_features.shape[-1] != conv_features.shape[-1]:
            spec_features = torch.nn.functional.interpolate(
                spec_features, size=conv_features.shape[-1], mode="linear", align_corners=False
            )

        fused = self.fusion(conv_features, spec_features)
        new_state = []
        for idx, block in enumerate(self.memory_blocks):
            block_state = state[idx] if state is not None and idx < len(state) else None
            fused, block_state = block(fused, block_state)
            new_state.append(block_state)

        conv_mask, spec_mask = self.mask_head(fused)
        bsz = conv_features.shape[0]
        t_enc = conv_features.shape[-1]
        conv_mask = conv_mask.view(bsz, self.num_sources, self.cfg.enc_channels, t_enc)
        spec_mask = spec_mask.view(bsz, self.num_sources, -1, t_enc)

        masked_conv = conv_mask * conv_features.unsqueeze(1)
        masked_spec = spec_mask * spec_features.unsqueeze(1)

        # Decode conv path
        conv_out = []
        for s in range(self.num_sources):
            decoded = self.conv_decoder(masked_conv[:, s])
            conv_out.append(decoded)
        conv_audio = torch.stack(conv_out, dim=1).squeeze(2)  # [B, S, T]

        # Decode spec path
        spec_out = []
        spec_t = spec_phase.shape[-1]
        for s in range(self.num_sources):
            spec_mag_s = masked_spec[:, s]
            if spec_mag_s.shape[-1] != spec_t:
                spec_mag_s = torch.nn.functional.interpolate(
                    spec_mag_s, size=spec_t, mode="linear", align_corners=False
                )
            decoded = self.spec_decoder(spec_mag_s, spec_phase, length=audio.shape[-1])
            spec_out.append(decoded)
        spec_audio = torch.stack(spec_out, dim=1)

        mixed = self.combiner(conv_audio, spec_audio)
        mixed = mixed[..., :orig_len]
        aux = {
            "conv_mask": conv_mask,
            "spec_mask": spec_mask,
            "state": new_state,
        }
        if return_aux:
            return mixed, aux
        return mixed
