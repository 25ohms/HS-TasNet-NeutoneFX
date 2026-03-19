from __future__ import annotations

import pathlib
from typing import Dict

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint, load_checkpoint_payload
from hs_tasnet.utils.audio import load_audio, save_audio
from hs_tasnet.utils.device import resolve_device

VALID_INPUT_CHANNEL_POLICIES = {"strict", "mono_downmix", "first_channel"}


def _resolve_model_config(cfg: Dict, checkpoint: str | None, device: torch.device) -> HSTasNetConfig:
    if checkpoint:
        checkpoint_payload = load_checkpoint_payload(checkpoint, map_location=device)
        checkpoint_cfg = checkpoint_payload.get("cfg", {})
        checkpoint_model_cfg = checkpoint_cfg.get("model")
        if checkpoint_model_cfg:
            return HSTasNetConfig(**checkpoint_model_cfg)
    return HSTasNetConfig(**cfg.get("model", {}))


def _align_input_channels(
    audio: torch.Tensor,
    expected_channels: int,
    channel_policy: str,
) -> torch.Tensor:
    if channel_policy not in VALID_INPUT_CHANNEL_POLICIES:
        raise ValueError(
            f"Unsupported infer.input_channel_policy '{channel_policy}'. "
            f"Expected one of {sorted(VALID_INPUT_CHANNEL_POLICIES)}."
        )

    actual_channels = audio.shape[1]
    if actual_channels == expected_channels:
        return audio

    if expected_channels == 1 and actual_channels > 1:
        if channel_policy == "strict":
            raise ValueError(
                "Input audio has multiple channels while the model expects mono. "
                "Set infer.input_channel_policy to 'mono_downmix' or 'first_channel'."
            )
        if channel_policy == "mono_downmix":
            return audio.mean(dim=1, keepdim=True)
        return audio[:, :1, :]

    raise ValueError(
        f"Input/model channel mismatch: input has {actual_channels} channel(s), "
        f"model expects {expected_channels}."
    )


def infer(cfg: Dict) -> pathlib.Path:
    infer_cfg = cfg.get("infer", {})
    input_path = infer_cfg.get("input_path")
    if not input_path:
        raise ValueError("infer.input_path must be set")

    output_dir = pathlib.Path(infer_cfg.get("output_dir", "runs/infer"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.get("device", {}).get("name"))
    checkpoint = infer_cfg.get("checkpoint")
    model_cfg = _resolve_model_config(cfg, checkpoint, device)
    model = HSTasNet(model_cfg)
    model.to(device)
    model.eval()

    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=device)

    audio_np, sr = load_audio(input_path, target_sr=model_cfg.sample_rate)
    audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # [1, C, T]
    audio = _align_input_channels(
        audio,
        expected_channels=model_cfg.audio_channels,
        channel_policy=infer_cfg.get("input_channel_policy", "mono_downmix"),
    )

    segment_seconds = infer_cfg.get("segment_seconds", 10.0)
    segment_samples = int(segment_seconds * sr)
    if audio.shape[-1] <= segment_samples:
        segments = [audio]
    else:
        segments = []
        for start in range(0, audio.shape[-1], segment_samples):
            end = min(start + segment_samples, audio.shape[-1])
            segments.append(audio[..., start:end])

    outputs = []
    with torch.no_grad():
        for seg in segments:
            pred = model(seg)
            outputs.append(pred.cpu())

    pred = torch.cat(outputs, dim=-1)  # [1, S, T]
    pred = pred.squeeze(0).numpy()

    stems = model_cfg.stems or ["drums", "bass", "vocals", "other"]
    for idx, stem in enumerate(stems):
        stem_audio = pred[idx]
        save_audio(output_dir / f"{stem}.wav", stem_audio, sr)

    return output_dir
