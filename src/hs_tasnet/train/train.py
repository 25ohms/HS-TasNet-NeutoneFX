from __future__ import annotations

import pathlib
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from hs_tasnet.data.collate import collate_examples
from hs_tasnet.data.datasets import AudioStemDataset, TinySyntheticDataset
from hs_tasnet.losses.waveform import l1_loss
from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint, save_checkpoint
from hs_tasnet.train.evaluate import evaluate
from hs_tasnet.train.optim import build_optimizer, build_scheduler
from hs_tasnet.utils.config import save_config
from hs_tasnet.utils.device import resolve_device
from hs_tasnet.utils.logging import init_tensorboard, log_config, setup_logger
from hs_tasnet.utils.seed import set_seed


def _build_dataset(cfg: Dict, split: str):
    data_cfg = cfg.get("data", {})
    segment_samples = int(data_cfg.get("segment_seconds", 4.0) * data_cfg.get("sample_rate", 44100))
    if data_cfg.get("tiny_dataset", False) or not data_cfg.get(f"{split}_dir"):
        return TinySyntheticDataset(
            length=8 if split == "train" else 4,
            segment_samples=segment_samples,
            num_stems=data_cfg.get("num_stems", 4),
            channels=data_cfg.get("audio_channels", 1),
        )
    return AudioStemDataset(
        root=data_cfg.get(f"{split}_dir"),
        stems=data_cfg.get("stems", ["drums", "bass", "vocals", "other"]),
        segment_samples=segment_samples,
        sample_rate=data_cfg.get("sample_rate", 44100),
    )


def train(cfg: Dict, resume: Optional[str] = None) -> pathlib.Path:
    logger = setup_logger()
    log_config(logger, cfg)
    set_seed(cfg.get("seed", 42))

    run_id = cfg.get("run", {}).get("id") or time.strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(cfg.get("run", {}).get("dir", "runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    device = resolve_device(cfg.get("device", {}).get("name"))
    model_cfg = HSTasNetConfig(**cfg.get("model", {}))
    model = HSTasNet(model_cfg).to(device)

    train_ds = _build_dataset(cfg, "train")
    val_ds = _build_dataset(cfg, "val")

    num_workers = cfg.get("data", {}).get("num_workers", 2)
    loader_kwargs = dict(
        train_ds,
        batch_size=cfg.get("train", {}).get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.get("data", {}).get("pin_memory", True),
        collate_fn=collate_examples,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.get("data", {}).get("prefetch_factor", 2)
    loader = DataLoader(**loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("val", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)

    writer = init_tensorboard(run_dir / "tb") if cfg.get("logging", {}).get("tensorboard", True) else None
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("device", {}).get("use_amp", False) and device.type == "cuda")

    epochs = cfg.get("train", {}).get("epochs", 1)
    grad_accum = cfg.get("train", {}).get("grad_accum_steps", 1)
    log_every = cfg.get("train", {}).get("log_every", 10)
    val_every = cfg.get("train", {}).get("val_every", 1)
    ckpt_every = cfg.get("train", {}).get("ckpt_every", 1)
    max_steps = cfg.get("train", {}).get("max_steps")

    model.train()
    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()
        for batch_idx, (mixture, stems) in enumerate(loader):
            mixture = mixture.to(device)
            stems = stems.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred, _ = model(mixture)
                loss = l1_loss(pred, stems)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            if global_step % log_every == 0:
                logger.info("train loss=%.4f", loss.item() * grad_accum, extra={"step": global_step})
                if writer:
                    writer.add_scalar("train/loss", loss.item() * grad_accum, global_step)

            global_step += 1
            if max_steps and global_step >= max_steps:
                break

        if (epoch + 1) % val_every == 0:
            metrics = evaluate(model, val_loader, device)
            logger.info("val l1=%.4f", metrics["l1"], extra={"step": global_step})
            if writer:
                writer.add_scalar("val/l1", metrics["l1"], global_step)

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch{epoch+1}.pt", model, optimizer, scheduler, epoch + 1, global_step, cfg)

        if max_steps and global_step >= max_steps:
            break

    if writer:
        writer.close()
    return run_dir
