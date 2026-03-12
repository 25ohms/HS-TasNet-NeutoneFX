#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
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


def _is_gcs_uri(path: str | None) -> bool:
    return bool(path and path.startswith("gs://"))


def _write_model_artifacts(
    run_dir: Path,
    latest_checkpoint: Path,
    output_dir: str | None,
    run_id: str,
) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{timestamp}_{run_id}"
    if not output_dir:
        output_path = Path("/tmp/aip_model")
        bundle_path = output_path / bundle_name
        bundle_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_checkpoint, bundle_path / "model.pt")
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            shutil.copy2(config_path, bundle_path / "config.yaml")
        return bundle_path.as_posix()

    if _is_gcs_uri(output_dir):
        with tempfile.TemporaryDirectory(prefix="hs_tasnet_model_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle_path = tmp_path / bundle_name
            bundle_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_checkpoint, bundle_path / "model.pt")
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                shutil.copy2(config_path, bundle_path / "config.yaml")
            sync_local_to_gcs(tmp_path, output_dir)
        return f"{output_dir.rstrip('/')}/{bundle_name}"

    output_path = Path(output_dir)
    bundle_path = output_path / bundle_name
    bundle_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_checkpoint, bundle_path / "model.pt")
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        shutil.copy2(config_path, bundle_path / "config.yaml")
    return bundle_path.as_posix()


def _write_run_artifacts(run_dir: Path, output_dir: str | None) -> None:
    if not output_dir:
        return

    if _is_gcs_uri(output_dir):
        sync_local_to_gcs(run_dir, output_dir)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, output_path, dirs_exist_ok=True)


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
    data_loader = cfg.get("data", {}).get("loader", "wav")
    overrides: List[str] = []
    overrides.extend(args.override or [])
    if data_loader == "musdb":
        overrides.extend(
            [
                f"data.musdb_root={args.data_dir}",
                "data.tiny_dataset=false",
                f"run.dir={args.run_dir}",
                f"run.id={run_id}",
            ]
        )
    else:
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

    latest = _latest_checkpoint(run_dir)
    model_bundle_uri = _write_model_artifacts(
        run_dir,
        latest,
        os.environ.get("AIP_MODEL_DIR"),
        run_id,
    )
    print(f"Model bundle saved: {model_bundle_uri}")

    if args.gcs_runs_uri:
        _write_run_artifacts(run_dir, f"{args.gcs_runs_uri.rstrip('/')}/{run_id}")
    else:
        _write_run_artifacts(run_dir, os.environ.get("AIP_CHECKPOINT_DIR"))


if __name__ == "__main__":
    main()
