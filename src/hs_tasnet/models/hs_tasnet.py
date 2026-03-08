from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from hs_tasnet.models.modules import (
    BranchSplit,
    ConvDecoder,
    ConvEncoder,
    DomainMaskHead,
    Fusion,
    HybridCombiner,
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
    enc_channels: int = 1024
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
        self.waveform_branch_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(self.cfg.enc_channels, self.cfg.lstm_hidden)
                for _ in range(self.cfg.lstm_layers)
            ]
        )
        self.spectral_branch_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(spec_channels, self.cfg.lstm_hidden)
                for _ in range(self.cfg.lstm_layers)
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(fused_channels, self.cfg.lstm_hidden)
                for _ in range(self.cfg.lstm_layers)
            ]
        )
        self.split = BranchSplit(fused_channels, self.cfg.enc_channels, spec_channels)
        self.conv_mask_head = DomainMaskHead(
            self.cfg.enc_channels,
            feature_channels=self.cfg.enc_channels,
            num_stems=self.cfg.num_stems,
            activation=self.cfg.mask_activation,
        )
        self.spec_mask_head = DomainMaskHead(
            spec_channels,
            feature_channels=spec_channels,
            num_stems=self.cfg.num_stems,
            activation=self.cfg.mask_activation,
        )
        self.combiner = HybridCombiner(alpha=0.5)

    def _run_block_stack(
        self,
        features: torch.Tensor,
        blocks: nn.ModuleList,
        state: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        new_state = []
        for idx, block in enumerate(blocks):
            block_state = state[idx] if state is not None and idx < len(state) else None
            features, block_state = block(features, block_state)
            new_state.append(block_state)
        return features, new_state

    def forward(
        self,
        audio: torch.Tensor,
        auto_curtail_length_to_multiple: bool = True,
        return_aux: bool = True,
        state: (
            Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]
            | List[Tuple[torch.Tensor, torch.Tensor]]
            | None
        ) = None,
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

        if isinstance(state, dict):
            waveform_state = state.get("waveform_branch")
            spectral_state = state.get("spectral_branch")
            shared_state = state.get("shared_branch")
        else:
            # Preserve compatibility with the earlier single-trunk state layout.
            waveform_state = None
            spectral_state = None
            shared_state = state

        waveform_features, waveform_state = self._run_block_stack(
            conv_features, self.waveform_branch_blocks, waveform_state
        )
        spectral_features, spectral_state = self._run_block_stack(
            spec_features, self.spectral_branch_blocks, spectral_state
        )
        fused = self.fusion(waveform_features, spectral_features)
        shared_features, shared_state = self._run_block_stack(
            fused, self.shared_blocks, shared_state
        )
        split_conv_features, split_spec_features = self.split(shared_features)

        conv_mask = self.conv_mask_head(split_conv_features)
        spec_mask = self.spec_mask_head(split_spec_features)
        bsz = conv_features.shape[0]
        t_enc = conv_features.shape[-1]
        conv_mask = conv_mask.view(bsz, self.num_sources, self.cfg.enc_channels, t_enc)
        spec_mask = spec_mask.view(bsz, self.num_sources, -1, t_enc)

        masked_conv = conv_mask * split_conv_features.unsqueeze(1)
        masked_spec = spec_mask * split_spec_features.unsqueeze(1)

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
            "state": {
                "waveform_branch": waveform_state,
                "spectral_branch": spectral_state,
                "shared_branch": shared_state,
            },
            "waveform_branch_features": waveform_features,
            "spectral_branch_features": spectral_features,
            "shared_features": shared_features,
            "split_conv_features": split_conv_features,
            "split_spec_features": split_spec_features,
        }
        if return_aux:
            return mixed, aux
        return mixed
