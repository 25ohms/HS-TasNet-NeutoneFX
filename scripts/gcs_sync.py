#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import storage


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def sync_gcs_to_local(gcs_uri: str, dest: Path) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    dest.mkdir(parents=True, exist_ok=True)
    blobs = client.list_blobs(bucket, prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)


def sync_local_to_gcs(src: Path, gcs_uri: str) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src).as_posix()
        blob_name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-uri", required=True, help="gs://bucket/prefix")
    parser.add_argument("--dest", required=True, help="Local directory")
    parser.add_argument(
        "--direction",
        choices=["download", "upload"],
        default="download",
        help="download: GCS -> local, upload: local -> GCS",
    )
    args = parser.parse_args()

    if args.direction == "download":
        sync_gcs_to_local(args.gcs_uri, Path(args.dest))
    else:
        sync_local_to_gcs(Path(args.dest), args.gcs_uri)


if __name__ == "__main__":
    main()
