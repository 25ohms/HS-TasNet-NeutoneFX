#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import List

from google.cloud import aiplatform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--staging-bucket", required=True, help="gs://bucket/staging")
    parser.add_argument("--container-uri", required=True)
    parser.add_argument("--model-serving-container", required=True)
    parser.add_argument("--job-display-name", default="hs-tasnet-train")
    parser.add_argument("--model-display-name", default="hs-tasnet-model")
    parser.add_argument("--base-output-dir", required=True, help="gs://bucket/model-artifacts")
    parser.add_argument("--machine-type", default="g2-standard-4")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--cfg", default="src/hs_tasnet/config/train.yaml")
    parser.add_argument("--dataset-uri", required=True, help="gs://bucket/musdb18")
    parser.add_argument("--gcs-runs-uri", default=None)
    parser.add_argument("--override", action="append")
    args = parser.parse_args()

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.job_display_name,
        container_uri=args.container_uri,
        model_serving_container_image_uri=args.model_serving_container,
    )

    worker_args: List[str] = [
        "python",
        "scripts/vertex_worker.py",
        "--cfg",
        args.cfg,
        "--dataset-uri",
        args.dataset_uri,
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
        base_output_dir=args.base_output_dir,
        service_account=args.service_account,
        network=args.network,
        args=worker_args,
    )

    print(f"Model registered: {model.resource_name}")


if __name__ == "__main__":
    main()
