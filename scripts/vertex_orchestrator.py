#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

import google.auth
from google.cloud import aiplatform


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
    parser.add_argument("--project", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--staging-bucket", default=None, help="gs://bucket/staging")
    parser.add_argument("--container-uri", default=None)
    parser.add_argument("--model-serving-container", default=None)
    parser.add_argument("--job-display-name", default="hs-tasnet-train")
    parser.add_argument("--model-display-name", default="hs-tasnet-model")
    parser.add_argument("--base-output-dir", default=None, help="gs://bucket/model-artifacts")
    parser.add_argument("--machine-type", default="g2-standard-4")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--cfg", default="src/hs_tasnet/config/train.yaml")
    parser.add_argument("--dataset-uri", default=None, help="gs://bucket/musdb18")
    parser.add_argument("--gcs-runs-uri", default=None)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()

    project = args.project or os.environ.get("PROJECT_ID")
    region = args.region or os.environ.get("REGION")
    staging_bucket = args.staging_bucket or os.environ.get("STAGING_BUCKET")
    container_uri = args.container_uri or os.environ.get("CONTAINER_URI")
    model_serving_container = args.model_serving_container or os.environ.get(
        "MODEL_SERVING_CONTAINER"
    )
    base_output_dir = args.base_output_dir or os.environ.get("BASE_OUTPUT_DIR")
    dataset_uri = args.dataset_uri or os.environ.get("DATASET_URI")
    service_account = args.service_account or os.environ.get("SERVICE_ACCOUNT")
    network = args.network or os.environ.get("NETWORK")

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
        display_name=args.job_display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container,
    )

    worker_args: List[str] = [
        "python",
        "scripts/vertex_worker.py",
        "--cfg",
        args.cfg,
        "--dataset-uri",
        dataset_uri,
    ]

    if args.gcs_runs_uri:
        worker_args.extend(["--gcs-runs-uri", args.gcs_runs_uri])

    if args.override:
        for ov in args.override:
            worker_args.extend(["--override", ov])

    model = job.run(
        model_display_name=args.model_display_name,
        replica_count=args.replica_count,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        base_output_dir=base_output_dir,
        service_account=service_account,
        network=network,
        args=worker_args,
    )

    print(f"Model registered: {model.resource_name}")


if __name__ == "__main__":
    main()
