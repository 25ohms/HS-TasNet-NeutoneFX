from __future__ import annotations

import pathlib
import time
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from hs_tasnet.data.collate import collate_examples
from hs_tasnet.data.datasets import AudioStemDataset, MusdbStemDataset, TinySyntheticDataset
from hs_tasnet.losses.metrics import compute_waveform_metrics
from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import (
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from hs_tasnet.train.evaluate import evaluate
from hs_tasnet.train.optim import build_optimizer, build_scheduler, get_optim_config
from hs_tasnet.train.regularization import compute_singular_value_penalty
from hs_tasnet.utils.config import save_config
from hs_tasnet.utils.device import resolve_device
from hs_tasnet.utils.logging import (
    init_tensorboard,
    log_config,
    maybe_init_wandb,
    setup_logger,
)
from hs_tasnet.utils.seed import set_seed


def _resolve_amp_enabled(cfg: Dict, device: torch.device) -> bool:
    use_amp = cfg.get("device", {}).get("use_amp")
    if use_amp is None:
        return device.type == "cuda"
    return bool(use_amp) and device.type == "cuda"


def _build_loader_kwargs(cfg: Dict) -> Dict:
    data_cfg = cfg.get("data", {})
    num_workers = int(data_cfg.get("num_workers", 2))
    loader_kwargs = {
        "batch_size": cfg.get("train", {}).get("batch_size", 4),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": data_cfg.get("pin_memory", True),
        "collate_fn": collate_examples,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = bool(
            data_cfg.get("persistent_workers", True)
        )
    return loader_kwargs


def _build_dataset(cfg: Dict, split: str):
    data_cfg = cfg.get("data", {})
    segment_samples = int(
        data_cfg.get("segment_seconds", 4.0) * data_cfg.get("sample_rate", 44100)
    )
    loader = data_cfg.get("loader", "wav")
    if loader == "musdb":
        musdb_root = data_cfg.get("musdb_root")
        if not musdb_root:
            raise ValueError("data.musdb_root must be set for loader=musdb")
        if split == "train":
            subset = "train"
            split_name = "train"
        else:
            subset = data_cfg.get("musdb_val_subset", "test")
            split_name = data_cfg.get("musdb_val_split", "valid") if subset == "train" else None
        return MusdbStemDataset(
            root=musdb_root,
            subset=subset,
            split=split_name,
            stems=data_cfg.get("stems", ["drums", "bass", "vocals", "other"]),
            segment_samples=segment_samples,
            sample_rate=data_cfg.get("sample_rate", 44100),
            audio_channels=data_cfg.get("audio_channels", 1),
            is_wav=bool(data_cfg.get("musdb_is_wav", False)),
            fallback_to_ratio_split=bool(data_cfg.get("musdb_fallback_to_ratio_split", True)),
            train_fraction=float(data_cfg.get("musdb_train_fraction", 0.8)),
            split_seed=int(data_cfg.get("musdb_split_seed", 42)),
        )
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


def train(
    cfg: Dict,
    resume: Optional[str] = None,
    epoch_end_callback: Optional[Callable[[int, int, pathlib.Path], None]] = None,
) -> pathlib.Path:
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

    loader = DataLoader(train_ds, **_build_loader_kwargs(cfg))
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("val", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    optim_cfg = get_optim_config(cfg)

    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)

    if cfg.get("logging", {}).get("tensorboard", True):
        writer = init_tensorboard(run_dir / "tb")
    else:
        writer = None
    wandb_run = maybe_init_wandb(cfg, run_id=run_id)
    scaler = torch.cuda.amp.GradScaler(enabled=_resolve_amp_enabled(cfg, device))

    epochs = cfg.get("train", {}).get("epochs", 1)
    grad_accum = cfg.get("train", {}).get("grad_accum_steps", 1)
    grad_clip_norm = optim_cfg.get("clip_grad", 0.0) or None
    train_metric_names = list(cfg.get("train", {}).get("metrics", ["l1"]))
    loss_name = str(optim_cfg.get("loss", "l1"))
    if loss_name not in train_metric_names:
        train_metric_names.append(loss_name)
    log_every = cfg.get("train", {}).get("log_every", 10)
    val_every = cfg.get("train", {}).get("val_every", 1)
    ckpt_every = cfg.get("train", {}).get("ckpt_every", 1)
    max_checkpoints = cfg.get("train", {}).get("max_checkpoints")
    max_batches = cfg.get("train", {}).get("max_batches")
    max_steps = cfg.get("train", {}).get("max_steps")
    sv_interval = max(
        1,
        int(cfg.get("regularization", {}).get("singular_value", {}).get("interval", 1)),
    )

    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            optimizer.zero_grad()
            for batch_idx, (mixture, stems) in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                mixture = mixture.to(device)
                stems = stems.to(device)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    pred = model(mixture)
                    train_metrics = compute_waveform_metrics(
                        pred,
                        stems,
                        metrics=train_metric_names,
                        target_channel_policy=cfg.get("train", {}).get(
                            "target_channel_policy", "strict"
                        ),
                    )
                    objective = train_metrics[loss_name]
                    if global_step % sv_interval == 0:
                        sv_penalty = compute_singular_value_penalty(model, cfg)
                    else:
                        sv_penalty = torch.zeros((), device=device)
                    loss = (objective + sv_penalty) / grad_accum

                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum == 0:
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler:
                        scheduler.step()

                if global_step % log_every == 0:
                    train_loss = loss.item() * grad_accum
                    reg_value = float(sv_penalty.detach().item())
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "train loss=%.4f objective=%.4f reg=%.4f lr=%.6f",
                        train_loss,
                        float(objective.detach().item()),
                        reg_value,
                        float(lr),
                        extra={"step": global_step},
                    )
                    if writer:
                        writer.add_scalar("train/loss", train_loss, global_step)
                        writer.add_scalar(
                            f"train/{loss_name}",
                            float(objective.detach().item()),
                            global_step,
                        )
                        writer.add_scalar(
                            "train/singular_value_penalty", reg_value, global_step
                        )
                        writer.add_scalar("train/lr", float(lr), global_step)
                        for name, value in train_metrics.items():
                            writer.add_scalar(
                                f"train_metrics/{name}",
                                float(value.detach().item()),
                                global_step,
                            )
                    if wandb_run:
                        wandb_run.log(
                            {
                                "train/loss": train_loss,
                                f"train/{loss_name}": float(objective.detach().item()),
                                "train/singular_value_penalty": reg_value,
                                "train/lr": float(lr),
                                **{
                                    f"train_metrics/{name}": float(value.detach().item())
                                    for name, value in train_metrics.items()
                                },
                                "step": global_step,
                            },
                            step=global_step,
                        )

                global_step += 1
                if max_steps and global_step >= max_steps:
                    break

            if (epoch + 1) % val_every == 0:
                metrics = evaluate(model, val_loader, device, eval_cfg=cfg.get("eval", {}))
                metric_log = " ".join(
                    f"val {name}={value:.4f}" for name, value in metrics.items()
                )
                logger.info(metric_log, extra={"step": global_step})
                if writer:
                    for name, value in metrics.items():
                        writer.add_scalar(f"val/{name}", value, global_step)
                if wandb_run:
                    wandb_run.log(
                        {f"val/{name}": value for name, value in metrics.items()},
                        step=global_step,
                    )

            if (epoch + 1) % ckpt_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoint_epoch{epoch+1}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    cfg,
                )
                prune_checkpoints(run_dir, max_checkpoints)

            if max_steps and global_step >= max_steps:
                break

            if epoch_end_callback is not None:
                epoch_end_callback(epoch + 1, global_step, run_dir)
    finally:
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()
    return run_dir
