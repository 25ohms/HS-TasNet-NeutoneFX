#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

import google.auth
from google.cloud import aiplatform

from scripts.orchestrator_config import get_nested, load_orchestrator_config


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
    parser.add_argument("--model-serving-container", default=None)
    parser.add_argument("--job-display-name", default=None)
    parser.add_argument("--model-display-name", default=None)
    parser.add_argument("--base-output-dir", default=None, help="gs://bucket/model-artifacts")
    parser.add_argument("--machine-type", default=None)
    parser.add_argument("--accelerator-type", default=None)
    parser.add_argument("--accelerator-count", type=int, default=None)
    parser.add_argument("--replica-count", type=int, default=None)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--dataset-uri", default=None, help="gs://bucket/musdb18")
    parser.add_argument("--gcs-runs-uri", default=None)
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
    model_serving_container = args.model_serving_container or os.environ.get(
        "MODEL_SERVING_CONTAINER"
    )
    if not model_serving_container:
        model_serving_container = get_nested(
            orchestrator_cfg, "vertex", "model_serving_container"
        )
    base_output_dir = (
        args.base_output_dir
        or get_nested(orchestrator_cfg, "vertex", "train", "base_output_dir")
        or os.environ.get("BASE_OUTPUT_DIR")
    )
    dataset_uri = (
        args.dataset_uri
        or get_nested(orchestrator_cfg, "vertex", "train", "dataset_uri")
        or os.environ.get("DATASET_URI")
    )
    service_account = (
        args.service_account
        or get_nested(orchestrator_cfg, "vertex", "service_account")
        or os.environ.get("SERVICE_ACCOUNT")
    )
    network = (
        args.network
        or get_nested(orchestrator_cfg, "vertex", "network")
        or os.environ.get("NETWORK")
    )
    job_display_name = args.job_display_name or get_nested(
        orchestrator_cfg, "vertex", "train", "job_display_name"
    ) or "hs-tasnet-train"
    model_display_name = args.model_display_name or get_nested(
        orchestrator_cfg, "vertex", "train", "model_display_name"
    ) or "hs-tasnet-model"
    worker_cfg = (
        args.cfg or get_nested(orchestrator_cfg, "vertex", "train", "cfg")
        or "src/hs_tasnet/config/train.yaml"
    )
    gcs_runs_uri = args.gcs_runs_uri or get_nested(
        orchestrator_cfg, "vertex", "train", "gcs_runs_uri"
    )
    machine_type = args.machine_type or get_nested(
        orchestrator_cfg, "vertex", "train", "machine_type"
    ) or "g2-standard-4"
    accelerator_type = args.accelerator_type or get_nested(
        orchestrator_cfg, "vertex", "train", "accelerator_type"
    ) or "NVIDIA_L4"
    accelerator_count = args.accelerator_count or get_nested(
        orchestrator_cfg, "vertex", "train", "accelerator_count"
    ) or 1
    replica_count = args.replica_count or get_nested(
        orchestrator_cfg, "vertex", "train", "replica_count"
    ) or 1

    missing = [
        name
        for name, value in [
            ("PROJECT_ID", project),
            ("REGION", region),
            ("STAGING_BUCKET", staging_bucket),
            ("CONTAINER_URI", container_uri),
            ("MODEL_SERVING_CONTAINER", model_serving_container),
            ("BASE_OUTPUT_DIR", base_output_dir),
            ("DATASET_URI", dataset_uri),
        ]
        if not value
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Missing required values (set flags or env): {missing_str}")

    _preflight_identity()
    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container,
    )

    worker_args: List[str] = [
        "python",
        "scripts/vertex_worker.py",
        "--cfg",
        worker_cfg,
        "--dataset-uri",
        dataset_uri,
    ]

    if gcs_runs_uri:
        worker_args.extend(["--gcs-runs-uri", gcs_runs_uri])

    if args.override:
        for ov in args.override:
            worker_args.extend(["--override", ov])

    model = job.run(
        model_display_name=model_display_name,
        replica_count=int(replica_count),
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=int(accelerator_count),
        base_output_dir=base_output_dir,
        service_account=service_account,
        network=network,
        args=worker_args,
    )

    print(f"Model registered: {model.resource_name}")


if __name__ == "__main__":
    main()
