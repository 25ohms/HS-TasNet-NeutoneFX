from __future__ import annotations

from argparse import Namespace

from torch.utils.data import DataLoader

from hs_tasnet.data.collate import collate_examples
from hs_tasnet.data.datasets import AudioStemDataset, TinySyntheticDataset
from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint
from hs_tasnet.train.evaluate import evaluate
from hs_tasnet.utils.config import apply_overrides, load_config
from hs_tasnet.utils.device import resolve_device
from hs_tasnet.utils.logging import setup_logger


def _build_dataset(cfg):
    data_cfg = cfg.get("data", {})
    segment_samples = int(data_cfg.get("segment_seconds", 4.0) * data_cfg.get("sample_rate", 44100))
    if data_cfg.get("tiny_dataset", False) or not data_cfg.get("val_dir"):
        return TinySyntheticDataset(
            length=4,
            segment_samples=segment_samples,
            num_stems=data_cfg.get("num_stems", 4),
            channels=data_cfg.get("audio_channels", 1),
        )
    return AudioStemDataset(
        root=data_cfg.get("val_dir"),
        stems=data_cfg.get("stems", ["drums", "bass", "vocals", "other"]),
        segment_samples=segment_samples,
        sample_rate=data_cfg.get("sample_rate", 44100),
    )

def run(args: Namespace) -> None:
    logger = setup_logger()
    cfg = load_config(args.cfg)
    cfg = apply_overrides(cfg, args.override or [])

    checkpoint = args.checkpoint or cfg.get("eval", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("eval.checkpoint or --checkpoint is required")

    device = resolve_device(cfg.get("device", {}).get("name"))
    model_cfg = HSTasNetConfig(**cfg.get("model", {}))
    model = HSTasNet(model_cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)

    dataset = _build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("eval", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
    )

    metrics = evaluate(model, loader, device)
    logger.info("Eval metrics: %s", metrics)
