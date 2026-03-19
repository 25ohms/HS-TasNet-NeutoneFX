#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any, Dict, List

import google.auth
from google.auth.transport.requests import AuthorizedSession

from hs_tasnet.utils.orchestrator_config import get_nested, load_orchestrator_config


def _preflight_identity() -> None:
    creds, adc_project = google.auth.default()
    quota_project = getattr(creds, "quota_project_id", None) or os.environ.get(
        "GOOGLE_CLOUD_QUOTA_PROJECT"
    )

    identity = None
    if hasattr(creds, "service_account_email"):
        identity = getattr(creds, "service_account_email")

    if not identity:
        try:
            identity = (
                subprocess.check_output(
                    [
                        "gcloud",
                        "auth",
                        "list",
                        "--filter=status:ACTIVE",
                        "--format=value(account)",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
                or None
            )
        except Exception:
            identity = None

    print("ADC project:", adc_project or "unknown")
    print("ADC identity:", identity or "unknown (user creds may hide email)")
    print("ADC quota project:", quota_project or "not set")


def _create_custom_job(
    *,
    project: str,
    region: str,
    display_name: str,
    container_uri: str,
    worker_args: List[str],
    machine_type: str,
    accelerator_type: str | None,
    accelerator_count: int,
    replica_count: int,
    boot_disk_size_gb: int,
    service_account: str | None,
    network: str | None,
    staging_bucket: str | None,
) -> Dict[str, Any]:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)

    machine_spec: Dict[str, Any] = {
        "machineType": machine_type,
    }
    if accelerator_type:
        machine_spec["acceleratorType"] = accelerator_type
        machine_spec["acceleratorCount"] = accelerator_count

    custom_job: Dict[str, Any] = {
        "displayName": display_name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": machine_spec,
                    "replicaCount": str(replica_count),
                    "diskSpec": {
                        "bootDiskType": "pd-ssd",
                        "bootDiskSizeGb": boot_disk_size_gb,
                    },
                    "containerSpec": {
                        "imageUri": container_uri,
                        "command": ["python", "scripts/vertex_eval_worker.py"],
                        "args": worker_args,
                    },
                }
            ]
        },
    }
    if service_account:
        custom_job["jobSpec"]["serviceAccount"] = service_account
    if network:
        custom_job["jobSpec"]["network"] = network
    if staging_bucket:
        custom_job["jobSpec"]["baseOutputDirectory"] = {
            "outputUriPrefix": staging_bucket.rstrip("/")
        }

    parent = f"projects/{project}/locations/{region}"
    url = f"https://{region}-aiplatform.googleapis.com/v1/{parent}/customJobs"
    response = session.post(url, json=custom_job, timeout=60)
    if not response.ok:
        raise RuntimeError(
            f"Vertex create custom job failed: {response.status_code} {response.text}"
        )
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orchestrator-cfg",
        default="config.yaml",
        help="Path to repo-level Vertex orchestration config",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--staging-bucket", default=None, help="gs://bucket/staging")
    parser.add_argument("--container-uri", default=None)
    parser.add_argument("--job-display-name", default=None)
    parser.add_argument("--machine-type", default=None)
    parser.add_argument("--accelerator-type", default=None)
    parser.add_argument("--accelerator-count", type=int, default=None)
    parser.add_argument("--replica-count", type=int, default=None)
    parser.add_argument("--boot-disk-size-gb", type=int, default=None)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--dataset-uri", default=None)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--eval-output-uri", default=None)
    parser.add_argument("--evaluation-display-name", default=None)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()

    orchestrator_cfg = load_orchestrator_config(args.orchestrator_cfg)

    project = (
        args.project
        or get_nested(orchestrator_cfg, "vertex", "project_id")
        or os.environ.get("PROJECT_ID")
    )
    region = (
        args.region
        or get_nested(orchestrator_cfg, "vertex", "region")
        or os.environ.get("REGION")
    )
    staging_bucket = (
        args.staging_bucket
        or get_nested(orchestrator_cfg, "vertex", "staging_bucket")
        or os.environ.get("STAGING_BUCKET")
    )
    container_uri = (
        args.container_uri
        or get_nested(orchestrator_cfg, "vertex", "container_uri")
        or os.environ.get("CONTAINER_URI")
    )
    dataset_uri = (
        args.dataset_uri
        or get_nested(orchestrator_cfg, "vertex", "eval", "dataset_uri")
        or os.environ.get("DATASET_URI")
    )
    model_uri = (
        args.model_uri
        or get_nested(orchestrator_cfg, "vertex", "eval", "model_uri")
        or os.environ.get("MODEL_URI")
    )
    eval_output_uri = (
        args.eval_output_uri
        or get_nested(orchestrator_cfg, "vertex", "eval", "eval_output_uri")
        or os.environ.get("EVAL_OUTPUT_URI")
    )
    service_account = (
        args.service_account
        or get_nested(orchestrator_cfg, "vertex", "service_account")
        or os.environ.get("SERVICE_ACCOUNT")
    )
    evaluation_display_name = (
        args.evaluation_display_name
        or get_nested(orchestrator_cfg, "vertex", "eval", "evaluation_display_name")
        or os.environ.get("EVAL_DISPLAY_NAME")
    )
    network = (
        args.network
        or get_nested(orchestrator_cfg, "vertex", "network")
        or os.environ.get("NETWORK")
    )
    job_display_name = args.job_display_name or get_nested(
        orchestrator_cfg, "vertex", "eval", "job_display_name"
    ) or "hs-tasnet-eval"
    worker_cfg = (
        args.cfg or get_nested(orchestrator_cfg, "vertex", "eval", "cfg")
        or "src/hs_tasnet/config/eval.yaml"
    )
    machine_type = args.machine_type or get_nested(
        orchestrator_cfg, "vertex", "eval", "machine_type"
    ) or "g2-standard-4"
    accelerator_type = args.accelerator_type or get_nested(
        orchestrator_cfg, "vertex", "eval", "accelerator_type"
    ) or "NVIDIA_L4"
    accelerator_count = args.accelerator_count or get_nested(
        orchestrator_cfg, "vertex", "eval", "accelerator_count"
    ) or 1
    replica_count = args.replica_count or get_nested(
        orchestrator_cfg, "vertex", "eval", "replica_count"
    ) or 1
    boot_disk_size_gb = args.boot_disk_size_gb or get_nested(
        orchestrator_cfg, "vertex", "eval", "boot_disk_size_gb"
    ) or 200

    missing = [
        name
        for name, value in [
            ("PROJECT_ID", project),
            ("REGION", region),
            ("STAGING_BUCKET", staging_bucket),
            ("CONTAINER_URI", container_uri),
            ("DATASET_URI", dataset_uri),
            ("MODEL_URI", model_uri),
            ("EVAL_OUTPUT_URI", eval_output_uri),
            ("SERVICE_ACCOUNT", service_account),
        ]
        if not value
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Missing required values (set flags or env): {missing_str}")

    _preflight_identity()

    worker_args: List[str] = [
        "--project",
        project,
        "--region",
        region,
        "--cfg",
        worker_cfg,
        "--dataset-uri",
        dataset_uri,
        "--model-uri",
        model_uri,
        "--eval-output-uri",
        eval_output_uri,
    ]
    if evaluation_display_name:
        worker_args.extend(["--evaluation-display-name", evaluation_display_name])
    if args.override:
        for override in args.override:
            worker_args.extend(["--override", override])

    job = _create_custom_job(
        project=project,
        region=region,
        display_name=job_display_name,
        container_uri=container_uri,
        worker_args=worker_args,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=int(accelerator_count),
        replica_count=int(replica_count),
        boot_disk_size_gb=int(boot_disk_size_gb),
        service_account=service_account,
        network=network,
        staging_bucket=staging_bucket,
    )

    print(f"Custom job submitted: {job['name']}")
    print(f"Evaluation artifacts target: {eval_output_uri}")


if __name__ == "__main__":
    main()
