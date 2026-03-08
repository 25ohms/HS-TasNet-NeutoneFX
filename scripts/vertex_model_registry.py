#!/usr/bin/env python
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import google.auth
from google.auth.transport.requests import AuthorizedSession


def _authorized_session() -> AuthorizedSession:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return AuthorizedSession(credentials)


def _wait_for_operation(
    session: AuthorizedSession,
    operation_name: str,
    region: str,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    operation_url = f"https://{region}-aiplatform.googleapis.com/v1/{operation_name}"
    while True:
        response = session.get(operation_url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("done"):
            if "error" in payload:
                raise RuntimeError(json.dumps(payload["error"], sort_keys=True))
            return payload
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for operation {operation_name}")
        time.sleep(poll_interval_seconds)


def import_model_evaluation(
    *,
    region: str,
    model_resource_name: str,
    evaluation_display_name: str,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    metrics_schema_uri: Optional[str] = None,
) -> Dict[str, Any]:
    session = _authorized_session()
    parent = model_resource_name.strip("/")
    url = f"https://{region}-aiplatform.googleapis.com/v1/{parent}/evaluations:import"

    model_evaluation: Dict[str, Any] = {
        "displayName": evaluation_display_name,
        "metrics": metrics,
        "metadata": metadata,
    }
    if metrics_schema_uri:
        model_evaluation["metricsSchemaUri"] = metrics_schema_uri

    response = session.post(
        url,
        json={"modelEvaluation": model_evaluation},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    operation_name = payload.get("name")
    if not operation_name:
        return payload

    return _wait_for_operation(session, operation_name, region=region)
