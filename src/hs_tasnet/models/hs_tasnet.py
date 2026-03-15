from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from hs_tasnet.models.contracts import BranchState, HSTasNetOutput
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
    build_group_norm,
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
    wave_lstm_hidden: int = 500
    spec_lstm_hidden: int = 500
    shared_lstm_hidden: int = 1000
    post_split_wave_lstm_hidden: int = 500
    post_split_spec_lstm_hidden: int = 500
    wave_num_blocks: int = 1
    spec_num_blocks: int = 1
    shared_num_blocks: int = 1
    post_split_num_blocks: int = 1
    fusion: str = "concat"
    mask_activation: str = "sigmoid"
    spec_mask_representation: str = "magnitude"
    bottleneck_group_norm_groups: int = 0


class HSTasNet(nn.Module):
    def __init__(self, cfg: HSTasNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or HSTasNetConfig()
        if self.cfg.fusion not in {"concat", "sum"}:
            raise ValueError(
                f"Unsupported fusion mode '{self.cfg.fusion}'. Expected 'concat' or 'sum'."
            )
        if self.cfg.stems is not None and len(self.cfg.stems) != self.cfg.num_stems:
            raise ValueError(
                f"model.stems length ({len(self.cfg.stems)}) must match model.num_stems "
                f"({self.cfg.num_stems})."
            )
        if self.cfg.spec_mask_representation != "magnitude":
            raise ValueError(
                "model.spec_mask_representation must be 'magnitude' for the current decoder path"
            )
        block_counts = {
            "wave_num_blocks": self.cfg.wave_num_blocks,
            "spec_num_blocks": self.cfg.spec_num_blocks,
            "shared_num_blocks": self.cfg.shared_num_blocks,
            "post_split_num_blocks": self.cfg.post_split_num_blocks,
        }
        for field_name, value in block_counts.items():
            if value < 0:
                raise ValueError(f"model.{field_name} must be >= 0")
        hidden_sizes = {
            "wave_lstm_hidden": self.cfg.wave_lstm_hidden,
            "spec_lstm_hidden": self.cfg.spec_lstm_hidden,
            "shared_lstm_hidden": self.cfg.shared_lstm_hidden,
            "post_split_wave_lstm_hidden": self.cfg.post_split_wave_lstm_hidden,
            "post_split_spec_lstm_hidden": self.cfg.post_split_spec_lstm_hidden,
        }
        for field_name, value in hidden_sizes.items():
            if value <= 0:
                raise ValueError(f"model.{field_name} must be > 0")
        if self.cfg.bottleneck_group_norm_groups < 0:
            raise ValueError("model.bottleneck_group_norm_groups must be >= 0")

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
                MemoryLSTMBlock(
                    self.cfg.enc_channels,
                    self.cfg.wave_lstm_hidden,
                    skip_mode="identity",
                )
                for _ in range(self.cfg.wave_num_blocks)
            ]
        )
        self.spectral_branch_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(
                    spec_channels,
                    self.cfg.spec_lstm_hidden,
                    skip_mode="identity",
                )
                for _ in range(self.cfg.spec_num_blocks)
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(
                    fused_channels,
                    self.cfg.shared_lstm_hidden,
                    skip_mode="identity",
                )
                for _ in range(self.cfg.shared_num_blocks)
            ]
        )
        self.split = BranchSplit(fused_channels, self.cfg.enc_channels, spec_channels)
        self.shared_norm = build_group_norm(
            fused_channels, self.cfg.bottleneck_group_norm_groups
        )
        self.split_conv_norm = build_group_norm(
            self.cfg.enc_channels, self.cfg.bottleneck_group_norm_groups
        )
        self.split_spec_norm = build_group_norm(
            spec_channels, self.cfg.bottleneck_group_norm_groups
        )
        self.post_split_wave_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(
                    self.cfg.enc_channels,
                    self.cfg.post_split_wave_lstm_hidden,
                    skip_mode="encoded",
                    skip_channels=self.cfg.enc_channels,
                )
                for _ in range(self.cfg.post_split_num_blocks)
            ]
        )
        self.post_split_spec_blocks = nn.ModuleList(
            [
                MemoryLSTMBlock(
                    spec_channels,
                    self.cfg.post_split_spec_lstm_hidden,
                    skip_mode="encoded",
                    skip_channels=spec_channels,
                )
                for _ in range(self.cfg.post_split_num_blocks)
            ]
        )
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
        self.combiner = HybridCombiner()

    def _run_block_stack(
        self,
        features: torch.Tensor,
        blocks: nn.ModuleList,
        state: (
            List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
            | None
        ) = None,
        encoded_representation: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        new_state = []
        for idx, block in enumerate(blocks):
            block_state = state[idx] if state is not None and idx < len(state) else None
            features, block_state = block(
                features,
                block_state,
                encoded_representation=encoded_representation,
            )
            new_state.append(block_state)
        return features, new_state

    def _decode_conv_batched(self, masked_conv: torch.Tensor) -> torch.Tensor:
        batch_size, num_stems, channels, frames = masked_conv.shape
        decoded = self.conv_decoder(
            masked_conv.reshape(batch_size * num_stems, channels, frames)
        )
        return decoded.reshape(batch_size, num_stems, self.audio_channels, -1).squeeze(2)

    def _apply_conv_frame_rescaling(
        self, masked_conv: torch.Tensor, frame_norms: torch.Tensor
    ) -> torch.Tensor:
        return masked_conv * frame_norms.unsqueeze(1)

    def _decode_spec_batched(
        self,
        masked_spec: torch.Tensor,
        spec_phase: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        batch_size, num_stems, spec_bins, spec_frames = masked_spec.shape
        target_frames = spec_phase.shape[-1]
        flat_spec = masked_spec.reshape(batch_size * num_stems, spec_bins, spec_frames)
        if spec_frames != target_frames:
            flat_spec = torch.nn.functional.interpolate(
                flat_spec, size=target_frames, mode="linear", align_corners=False
            )
        flat_phase = (
            spec_phase.unsqueeze(1)
            .expand(batch_size, num_stems, spec_bins, target_frames)
            .reshape(batch_size * num_stems, spec_bins, target_frames)
        )
        decoded = self.spec_decoder(flat_spec, flat_phase, length=audio_length)
        return decoded.reshape(batch_size, num_stems, -1)

    def forward(
        self,
        audio: torch.Tensor,
        auto_curtail_length_to_multiple: bool = True,
        state: BranchState | None = None,
    ) -> torch.Tensor:
        return self.separate(
            audio=audio,
            auto_curtail_length_to_multiple=auto_curtail_length_to_multiple,
            state=state,
        ).audio

    def separate(
        self,
        audio: torch.Tensor,
        auto_curtail_length_to_multiple: bool = True,
        state: BranchState | None = None,
    ) -> HSTasNetOutput:
        # audio: [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.shape[1] != self.audio_channels:
            raise ValueError(
                f"Expected {self.audio_channels} audio channels, got {audio.shape[1]}"
            )
        orig_len = audio.shape[-1]
        if auto_curtail_length_to_multiple:
            audio = pad_to_multiple(audio, self.cfg.hop_size)

        conv_features, conv_frame_norms = self.conv_encoder.encode_with_norms(
            audio
        )  # [B, Cc, Tenc]
        spec_mag, spec_phase = self.spec_encoder(audio)  # [B, F, Tspec]
        spec_features = spec_mag
        if spec_features.shape[-1] != conv_features.shape[-1]:
            spec_features = torch.nn.functional.interpolate(
                spec_features, size=conv_features.shape[-1], mode="linear", align_corners=False
            )

        if state is None:
            waveform_state = None
            spectral_state = None
            shared_state = None
            post_split_wave_state = None
            post_split_spec_state = None
        else:
            waveform_state = state.get("waveform_branch")
            spectral_state = state.get("spectral_branch")
            shared_state = state.get("shared_branch")
            post_split_wave_state = state.get("post_split_wave_branch")
            post_split_spec_state = state.get("post_split_spec_branch")

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
        shared_features = self.shared_norm(shared_features)
        split_conv_features, split_spec_features = self.split(shared_features)
        split_conv_features = self.split_conv_norm(split_conv_features)
        split_spec_features = self.split_spec_norm(split_spec_features)
        split_conv_features, post_split_wave_state = self._run_block_stack(
            split_conv_features,
            self.post_split_wave_blocks,
            post_split_wave_state,
            encoded_representation=conv_features,
        )
        split_spec_features, post_split_spec_state = self._run_block_stack(
            split_spec_features,
            self.post_split_spec_blocks,
            post_split_spec_state,
            encoded_representation=spec_features,
        )

        conv_mask = self.conv_mask_head(split_conv_features)
        spec_mask = self.spec_mask_head(split_spec_features)
        bsz = conv_features.shape[0]
        t_enc = conv_features.shape[-1]
        conv_mask = conv_mask.view(bsz, self.num_sources, self.cfg.enc_channels, t_enc)
        spec_mask = spec_mask.view(bsz, self.num_sources, -1, t_enc)

        # Predict masks from post-split memory features, then apply them to the
        # encoder-domain representations.
        masked_conv = conv_mask * conv_features.unsqueeze(1)
        masked_conv = self._apply_conv_frame_rescaling(masked_conv, conv_frame_norms)
        masked_spec = spec_mask * spec_features.unsqueeze(1)

        # Decode conv path
        conv_audio = self._decode_conv_batched(masked_conv)  # [B, S, T]

        # Decode spec path
        spec_audio = self._decode_spec_batched(masked_spec, spec_phase, audio.shape[-1])

        mixed = self.combiner(conv_audio, spec_audio)
        conv_audio = conv_audio[..., :orig_len]
        spec_audio = spec_audio[..., :orig_len]
        mixed = mixed[..., :orig_len]
        next_state: BranchState = {
            "waveform_branch": waveform_state,
            "spectral_branch": spectral_state,
            "shared_branch": shared_state,
            "post_split_wave_branch": post_split_wave_state,
            "post_split_spec_branch": post_split_spec_state,
        }
        return HSTasNetOutput(
            audio=mixed,
            conv_audio=conv_audio,
            spec_audio=spec_audio,
            conv_mask=conv_mask,
            spec_mask=spec_mask,
            state=next_state,
            waveform_branch_features=waveform_features,
            spectral_branch_features=spectral_features,
            shared_features=shared_features,
            split_conv_features=split_conv_features,
            split_spec_features=split_spec_features,
        )

    def init_stream_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Dict[str, object]:
        target_device = device or next(self.parameters()).device
        target_dtype = dtype or next(self.parameters()).dtype
        overlap = self.cfg.window_size - self.cfg.hop_size
        if overlap <= 0:
            raise ValueError("Streaming requires model.window_size > model.hop_size")
        if overlap != self.cfg.hop_size:
            raise ValueError(
                "Streaming overlap-add currently requires window_size == 2 * hop_size"
            )
        buffer = torch.zeros(
            batch_size,
            self.audio_channels,
            self.cfg.window_size,
            device=target_device,
            dtype=target_dtype,
        )
        conv_overlap = torch.zeros(
            batch_size,
            self.num_sources,
            overlap,
            device=target_device,
            dtype=target_dtype,
        )
        spec_overlap = torch.zeros(
            batch_size,
            self.num_sources,
            overlap,
            device=target_device,
            dtype=target_dtype,
        )
        return {
            "buffer": buffer,
            "branch_state": None,
            "conv_overlap": conv_overlap,
            "spec_overlap": spec_overlap,
        }

    @torch.no_grad()
    def stream_step(
        self, x_hop: torch.Tensor, stream_state: Dict[str, object]
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if x_hop.dim() == 2:
            x_hop = x_hop.unsqueeze(0)
        if x_hop.shape[-1] != self.cfg.hop_size:
            raise ValueError(f"Expected hop size {self.cfg.hop_size}, got {x_hop.shape[-1]}")
        if x_hop.shape[1] != self.audio_channels:
            raise ValueError("Channel mismatch for streaming input")

        buffer = stream_state["buffer"]
        if not isinstance(buffer, torch.Tensor):
            raise TypeError("stream_state['buffer'] must be a tensor")
        branch_state = stream_state.get("branch_state")
        if branch_state is not None and not isinstance(branch_state, dict):
            raise TypeError("stream_state['branch_state'] must be a branch state dictionary")
        conv_overlap = stream_state.get("conv_overlap")
        spec_overlap = stream_state.get("spec_overlap")
        if not isinstance(conv_overlap, torch.Tensor):
            raise TypeError("stream_state['conv_overlap'] must be a tensor")
        if not isinstance(spec_overlap, torch.Tensor):
            raise TypeError("stream_state['spec_overlap'] must be a tensor")

        x_hop = x_hop.to(device=buffer.device, dtype=buffer.dtype)
        buffer = torch.cat([buffer[..., self.cfg.hop_size :], x_hop], dim=-1)
        output = self.separate(
            buffer,
            auto_curtail_length_to_multiple=False,
            state=branch_state,
        )
        overlap = self.cfg.window_size - self.cfg.hop_size
        conv_hop = conv_overlap + output.conv_audio[..., :overlap]
        spec_hop = spec_overlap + output.spec_audio[..., :overlap]
        mixed_hop = self.combiner(conv_hop, spec_hop)

        updated_state = {
            "buffer": buffer,
            "branch_state": output.state,
            "conv_overlap": output.conv_audio[..., overlap:],
            "spec_overlap": output.spec_audio[..., overlap:],
        }
        return mixed_hop, updated_state
