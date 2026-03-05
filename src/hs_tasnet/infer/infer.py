from __future__ import annotations

import pathlib
from typing import Dict

import torch

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint
from hs_tasnet.utils.audio import load_audio, save_audio
from hs_tasnet.utils.device import resolve_device


def infer(cfg: Dict) -> pathlib.Path:
    infer_cfg = cfg.get("infer", {})
    input_path = infer_cfg.get("input_path")
    if not input_path:
        raise ValueError("infer.input_path must be set")

    output_dir = pathlib.Path(infer_cfg.get("output_dir", "runs/infer"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = HSTasNetConfig(**cfg.get("model", {}))
    model = HSTasNet(model_cfg)
    device = resolve_device(cfg.get("device", {}).get("name"))
    model.to(device)
    model.eval()

    checkpoint = infer_cfg.get("checkpoint")
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=device)

    audio_np, sr = load_audio(input_path, target_sr=model_cfg.sample_rate)
    audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # [1, C, T]

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
            pred, _ = model(seg)
            outputs.append(pred.cpu())

    pred = torch.cat(outputs, dim=-1)  # [1, S, T]
    pred = pred.squeeze(0).numpy()

    stems = model_cfg.stems or ["drums", "bass", "vocals", "other"]
    for idx, stem in enumerate(stems):
        stem_audio = pred[idx]
        save_audio(output_dir / f"{stem}.wav", stem_audio, sr)

    return output_dir
