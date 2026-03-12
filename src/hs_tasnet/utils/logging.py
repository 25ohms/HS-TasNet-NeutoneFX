from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any, Dict

import yaml


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        base = f"{ts} | {record.levelname:<7} | {record.getMessage()}"
        if record.__dict__.get("step") is not None:
            base += f" | step={record.__dict__['step']}"
        return base


def setup_logger(name: str = "hs_tasnet", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_StructuredFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_config(logger: logging.Logger, cfg: Dict[str, Any]) -> None:
    dumped = yaml.safe_dump(cfg, sort_keys=False)
    logger.info("Resolved config:\n%s", dumped)


def init_tensorboard(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional in runtime
        raise RuntimeError("tensorboard is not available") from exc
    return SummaryWriter(log_dir=log_dir)


def maybe_init_wandb(cfg: Dict[str, Any], run_id: str | None = None):
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("wandb is enabled but not installed") from exc

    try:
        return wandb.init(
            project=wandb_cfg.get("project", "hs-tasnet"),
            entity=wandb_cfg.get("entity"),
            name=run_id,
            config=cfg,
        )
    except Exception as exc:
        msg = str(exc)
        if "No API key configured" in msg:
            raise RuntimeError(
                "W&B is enabled but no API key is available in this runtime. "
                "Set WANDB_API_KEY in the worker environment (for Vertex, export it "
                "before invoking hs_tasnet.vertex_orchestrator so it can be forwarded)."
            ) from exc
        raise
