from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from hs_tasnet.utils.orchestrator_config import get_nested, load_orchestrator_config


def _load_vertex_worker_module():
    worker_path = Path(__file__).resolve().parents[1] / "scripts" / "vertex_worker.py"
    spec = importlib.util.spec_from_file_location("vertex_worker_for_test", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_orchestrator_config_helpers(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "vertex:\n"
        "  project_id: realtime-stems\n"
        "  eval:\n"
        "    model_uri: gs://bucket/model\n",
        encoding="utf-8",
    )

    cfg = load_orchestrator_config(cfg_path)

    assert get_nested(cfg, "vertex", "project_id") == "realtime-stems"
    assert get_nested(cfg, "vertex", "eval", "model_uri") == "gs://bucket/model"
    assert get_nested(cfg, "vertex", "missing") is None


def test_vertex_script_modules_import():
    pytest.importorskip("google.auth")

    eval_module = importlib.import_module("hs_tasnet.vertex_eval_orchestrator")
    train_module = importlib.import_module("hs_tasnet.vertex_orchestrator")

    assert hasattr(eval_module, "main")
    assert hasattr(train_module, "main")


def test_vertex_worker_periodic_sync_callback(monkeypatch, tmp_path: Path):
    worker_module = _load_vertex_worker_module()

    synced = []

    def fake_sync_local_to_gcs(src: Path, gcs_uri: str) -> None:
        synced.append((src, gcs_uri))

    monkeypatch.setattr(worker_module, "sync_gcs_to_local", lambda _uri, _dest: None)
    monkeypatch.setattr(worker_module, "sync_local_to_gcs", fake_sync_local_to_gcs)

    callback = worker_module._build_periodic_sync_callback("gs://bucket/runs/job-1", 2)
    assert callback is not None

    callback(1, 10, tmp_path)
    callback(2, 20, tmp_path)

    assert synced == [(tmp_path, "gs://bucket/runs/job-1")]


def test_vertex_worker_periodic_sync_callback_rejects_invalid_interval():
    worker_module = _load_vertex_worker_module()

    with pytest.raises(ValueError):
        worker_module._build_periodic_sync_callback("gs://bucket/runs/job-1", 0)
