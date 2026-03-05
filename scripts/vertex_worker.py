#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from hs_tasnet.train.train import train
from hs_tasnet.utils.config import apply_overrides, load_config

try:
    from scripts.gcs_sync import sync_gcs_to_local, sync_local_to_gcs
except Exception:
    # Fallback if scripts isn't on the import path
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "gcs_sync", str(Path(__file__).parent / "gcs_sync.py")
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["gcs_sync"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    sync_gcs_to_local = module.sync_gcs_to_local  # type: ignore
    sync_local_to_gcs = module.sync_local_to_gcs  # type: ignore


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--dataset-uri", default=None, help="gs://bucket/prefix")
    parser.add_argument("--data-dir", default="/mnt/data/musdb18")
    parser.add_argument("--run-dir", default="/mnt/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gcs-runs-uri", default=None)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()

    run_id = args.run_id or os.environ.get("AIP_JOB_NAME") or "vertex-run"

    if args.dataset_uri:
        sync_gcs_to_local(args.dataset_uri, Path(args.data_dir))

    cfg = load_config(args.cfg)
    overrides: List[str] = []
    overrides.extend(args.override or [])
    overrides.extend(
        [
            f"data.train_dir={args.data_dir}/train",
            f"data.val_dir={args.data_dir}/test",
            "data.tiny_dataset=false",
            f"run.dir={args.run_dir}",
            f"run.id={run_id}",
        ]
    )
    cfg = apply_overrides(cfg, overrides)

    run_dir = train(cfg, resume=args.resume)

    model_dir = Path(os.environ.get("AIP_MODEL_DIR", "/tmp/aip_model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_checkpoint(run_dir)
    shutil.copy2(latest, model_dir / "model.pt")

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        shutil.copy2(config_path, model_dir / "config.yaml")

    if args.gcs_runs_uri:
        sync_local_to_gcs(run_dir, f"{args.gcs_runs_uri.rstrip('/')}/{run_id}")


if __name__ == "__main__":
    main()
