from __future__ import annotations

import importlib

import pytest

from hs_tasnet.utils.orchestrator_config import get_nested, load_orchestrator_config


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
