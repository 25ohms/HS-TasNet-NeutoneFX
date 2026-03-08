#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from hs_tasnet.data.datasets import AudioStemDataset, MusdbStemDataset, TinySyntheticDataset
from hs_tasnet.losses.waveform import l1_loss
from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint
from hs_tasnet.utils.config import apply_overrides, load_config, save_config
from hs_tasnet.utils.device import resolve_device
from hs_tasnet.utils.logging import log_config, setup_logger
from hs_tasnet.utils.seed import set_seed

try:
    from scripts.gcs_sync import sync_gcs_to_local, sync_local_to_gcs
    from scripts.vertex_model_registry import import_model_evaluation
except Exception:
    import importlib.util
    import sys

    base_dir = Path(__file__).parent

    gcs_spec = importlib.util.spec_from_file_location("gcs_sync", str(base_dir / "gcs_sync.py"))
    gcs_module = importlib.util.module_from_spec(gcs_spec)  # type: ignore[arg-type]
    sys.modules["gcs_sync"] = gcs_module
    assert gcs_spec and gcs_spec.loader
    gcs_spec.loader.exec_module(gcs_module)  # type: ignore[union-attr]
    sync_gcs_to_local = gcs_module.sync_gcs_to_local  # type: ignore[attr-defined]
    sync_local_to_gcs = gcs_module.sync_local_to_gcs  # type: ignore[attr-defined]

    registry_spec = importlib.util.spec_from_file_location(
        "vertex_model_registry", str(base_dir / "vertex_model_registry.py")
    )
    registry_module = importlib.util.module_from_spec(registry_spec)  # type: ignore[arg-type]
    sys.modules["vertex_model_registry"] = registry_module
    assert registry_spec and registry_spec.loader
    registry_spec.loader.exec_module(registry_module)  # type: ignore[union-attr]
    import_model_evaluation = registry_module.import_model_evaluation  # type: ignore[attr-defined]


def _is_gcs_uri(path: str | None) -> bool:
    return bool(path and path.startswith("gs://"))


def _prepare_local_copy(src: str, dest: Path) -> Path:
    if _is_gcs_uri(src):
        sync_gcs_to_local(src, dest)
        return dest
    source_path = Path(src)
    if source_path.is_dir():
        shutil.copytree(source_path, dest, dirs_exist_ok=True)
        return dest
    raise FileNotFoundError(f"Expected a directory at {src}")


def _load_runtime_config(
    model_dir: Path, fallback_cfg: str, overrides: Iterable[str]
) -> Dict[str, Any]:
    config_path = model_dir / "config.yaml"
    cfg_path = config_path if config_path.exists() else Path(fallback_cfg)
    cfg = load_config(cfg_path)
    return apply_overrides(cfg, overrides)


def _segment_samples(cfg: Dict[str, Any]) -> int:
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get("segment_seconds", 4.0) * data_cfg.get("sample_rate", 44100))


def _build_dataset(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    segment_samples = _segment_samples(cfg)
    loader = data_cfg.get("loader", "wav")
    if loader == "musdb":
        musdb_root = data_cfg.get("musdb_root")
        if not musdb_root:
            raise ValueError("data.musdb_root must be set for loader=musdb")
        subset = data_cfg.get("musdb_val_subset", "test")
        split = data_cfg.get("musdb_val_split", "valid") if subset == "train" else None
        return MusdbStemDataset(
            root=musdb_root,
            subset=subset,
            split=split,
            stems=data_cfg.get("stems", ["drums", "bass", "vocals", "other"]),
            segment_samples=segment_samples,
            sample_rate=data_cfg.get("sample_rate", 44100),
            audio_channels=data_cfg.get("audio_channels", 1),
            is_wav=bool(data_cfg.get("musdb_is_wav", False)),
        )
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


def _audio_example_from_track(dataset: Any, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
    if isinstance(dataset, TinySyntheticDataset):
        example = dataset[index]
        return f"synthetic_{index:04d}", example.mixture, example.stems

    if isinstance(dataset, AudioStemDataset):
        track_dir = dataset.tracks[index]
        mixture, stem_audio = dataset._load_track(track_dir)
        stems = np.stack(stem_audio, axis=0)
        return (
            track_dir.name,
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(stems.astype(np.float32)),
        )

    if isinstance(dataset, MusdbStemDataset):
        track = dataset.tracks[index]
        mixture = dataset._mix_channels(track.audio).T
        stem_audio = []
        for stem in dataset.stems:
            audio = dataset._mix_channels(track.targets[stem].audio)
            stem_audio.append(audio)
        stems = np.stack(stem_audio, axis=0).transpose(0, 2, 1)
        return (
            getattr(track, "name", f"track_{index:04d}"),
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(stems.astype(np.float32)),
        )

    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def _evaluate_dataset(
    model: torch.nn.Module, dataset: Any, device: torch.device
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    total_l1 = 0.0

    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            track_id, mixture, stems = _audio_example_from_track(dataset, index)
            mixture = mixture.unsqueeze(0).to(device)
            stems = stems.unsqueeze(0).to(device)
            pred, _ = model(mixture)
            loss = float(l1_loss(pred, stems).item())
            total_l1 += loss
            rows.append(
                {
                    "track_id": track_id,
                    "metrics": {
                        "l1": loss,
                    },
                }
            )
    model.train()

    metrics = {
        "l1": total_l1 / max(len(rows), 1),
        "num_examples": float(len(rows)),
    }
    return metrics, rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--model-resource-name", required=True)
    parser.add_argument("--eval-output-uri", required=True)
    parser.add_argument("--cfg", default="src/hs_tasnet/config/eval.yaml")
    parser.add_argument("--data-dir", default="/mnt/data/musdb18")
    parser.add_argument("--model-dir", default="/mnt/model")
    parser.add_argument("--output-dir", default="/mnt/eval")
    parser.add_argument("--evaluation-display-name", default=None)
    parser.add_argument("--metrics-schema-uri", default=None)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()

    logger = setup_logger()
    run_id = os.environ.get("AIP_JOB_NAME") or f"vertex-eval-{int(time.time())}"
    local_model_dir = _prepare_local_copy(args.model_uri, Path(args.model_dir))
    local_data_dir = _prepare_local_copy(args.dataset_uri, Path(args.data_dir))

    local_output_dir = Path(args.output_dir) / run_id
    local_output_dir.mkdir(parents=True, exist_ok=True)

    data_loader = None
    base_overrides: List[str] = [
        "data.tiny_dataset=false",
        f"run.id={run_id}",
        f"run.dir={local_output_dir.as_posix()}",
    ]

    checkpoint_path = local_model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

    cfg_probe = _load_runtime_config(local_model_dir, args.cfg, [])
    data_loader = cfg_probe.get("data", {}).get("loader", "wav")
    if data_loader == "musdb":
        base_overrides.append(f"data.musdb_root={local_data_dir.as_posix()}")
    else:
        base_overrides.append(f"data.val_dir={(local_data_dir / 'test').as_posix()}")

    cfg = _load_runtime_config(local_model_dir, args.cfg, [*base_overrides, *(args.override or [])])
    save_config(cfg, local_output_dir / "resolved_config.yaml")
    log_config(logger, cfg)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", {}).get("name"))
    model_cfg = HSTasNetConfig(**cfg.get("model", {}))
    model = HSTasNet(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device, restore_rng=False)

    dataset = _build_dataset(cfg)
    metrics, rows = _evaluate_dataset(model, dataset, device)

    metrics_payload = {
        "checkpoint_path": checkpoint_path.as_posix(),
        "dataset_uri": args.dataset_uri,
        "loader": data_loader,
        "metrics": metrics,
    }
    _write_json(local_output_dir / "metrics.json", metrics_payload)
    _write_jsonl(local_output_dir / "row_metrics.jsonl", rows)

    sync_local_to_gcs(local_output_dir, args.eval_output_uri)
    row_metrics_uri = f"{args.eval_output_uri.rstrip('/')}/row_metrics.jsonl"

    evaluation_display_name = args.evaluation_display_name or run_id
    metadata = {
        "evaluationDatasetType": data_loader,
        "evaluationDatasetPath": args.dataset_uri,
        "checkpointUri": args.model_uri,
        "rowMetricsUri": row_metrics_uri,
        "runId": run_id,
    }
    import_result = import_model_evaluation(
        region=args.region,
        model_resource_name=args.model_resource_name,
        evaluation_display_name=evaluation_display_name,
        metrics=metrics,
        metadata=metadata,
        metrics_schema_uri=args.metrics_schema_uri,
    )
    _write_json(local_output_dir / "model_registry_import.json", import_result)
    sync_local_to_gcs(local_output_dir, args.eval_output_uri)

    logger.info("Eval metrics: %s", metrics)
    logger.info("Vertex model evaluation import completed for %s", args.model_resource_name)


if __name__ == "__main__":
    main()
