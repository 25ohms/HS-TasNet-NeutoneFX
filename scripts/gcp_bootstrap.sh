#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a GCP VM for training on GCS-backed datasets.
# Assumes gcloud/gsutil is installed and credentials are available.

GCS_BUCKET=${GCS_BUCKET:-""}
DATASET_PREFIX=${DATASET_PREFIX:-"musdb18"}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-"/mnt/data/${DATASET_PREFIX}"}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"/mnt/artifacts"}
RUNS_DIR=${RUNS_DIR:-"/mnt/runs"}
SYNC_MODE=${SYNC_MODE:-"rsync"} # rsync | mount

if [[ -z "${GCS_BUCKET}" ]]; then
  echo "GCS_BUCKET is required (e.g. GCS_BUCKET=gs://my-bucket)" >&2
  exit 1
fi

mkdir -p "${LOCAL_DATA_DIR}" "${ARTIFACTS_DIR}" "${RUNS_DIR}"

if [[ "${SYNC_MODE}" == "mount" ]]; then
  if ! command -v gcsfuse >/dev/null 2>&1; then
    echo "gcsfuse not found. Install it or set SYNC_MODE=rsync." >&2
    exit 1
  fi
  sudo mkdir -p "${LOCAL_DATA_DIR}"
  sudo gcsfuse "${GCS_BUCKET#gs://}" "${LOCAL_DATA_DIR}" >/dev/null
  echo "Mounted ${GCS_BUCKET} at ${LOCAL_DATA_DIR}"
else
  echo "Syncing ${GCS_BUCKET}/${DATASET_PREFIX} -> ${LOCAL_DATA_DIR}"
  gsutil -m rsync -r "${GCS_BUCKET}/${DATASET_PREFIX}" "${LOCAL_DATA_DIR}"
fi

# Optional: pull any existing artifacts
if gsutil ls "${GCS_BUCKET}/artifacts" >/dev/null 2>&1; then
  echo "Syncing artifacts from ${GCS_BUCKET}/artifacts -> ${ARTIFACTS_DIR}"
  gsutil -m rsync -r "${GCS_BUCKET}/artifacts" "${ARTIFACTS_DIR}"
fi

# Optional: pull prior runs
if gsutil ls "${GCS_BUCKET}/runs" >/dev/null 2>&1; then
  echo "Syncing runs from ${GCS_BUCKET}/runs -> ${RUNS_DIR}"
  gsutil -m rsync -r "${GCS_BUCKET}/runs" "${RUNS_DIR}"
fi

echo "Bootstrap complete."
