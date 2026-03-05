#!/usr/bin/env bash
set -euo pipefail

# Submits a training job from the orchestrator VM.
# Runs locally on the VM after ensuring data/artifacts are synced.

GCS_BUCKET=${GCS_BUCKET:-""}
DATASET_PREFIX=${DATASET_PREFIX:-"musdb18"}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-"/mnt/data/${DATASET_PREFIX}"}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"/mnt/artifacts"}
RUNS_DIR=${RUNS_DIR:-"/mnt/runs"}
JOB_NAME=${JOB_NAME:-"hs-tasnet-$(date +%Y%m%d-%H%M%S)"}
CFG=${CFG:-"src/hs_tasnet/config/train.yaml"}
SYNC_MODE=${SYNC_MODE:-"rsync"}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}

if [[ -z "${GCS_BUCKET}" ]]; then
  echo "GCS_BUCKET is required (e.g. GCS_BUCKET=gs://my-bucket)" >&2
  exit 1
fi

# Ensure data + artifacts are available
GCS_BUCKET="${GCS_BUCKET}" \
DATASET_PREFIX="${DATASET_PREFIX}" \
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" \
ARTIFACTS_DIR="${ARTIFACTS_DIR}" \
RUNS_DIR="${RUNS_DIR}" \
SYNC_MODE="${SYNC_MODE}" \
scripts/gcp_bootstrap.sh

RUN_DIR="${RUNS_DIR}/${JOB_NAME}"
mkdir -p "${RUN_DIR}"

OVERRIDES=(
  "data.train_dir=${LOCAL_DATA_DIR}/train"
  "data.val_dir=${LOCAL_DATA_DIR}/test"
  "data.tiny_dataset=false"
  "run.dir=${RUNS_DIR}"
  "run.id=${JOB_NAME}"
)

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  while read -r line; do
    [[ -n "${line}" ]] && OVERRIDES+=("${line}")
  done <<< "${EXTRA_OVERRIDES}"
fi

CMD=("hs-tasnet" "train" "--cfg" "${CFG}")
for ov in "${OVERRIDES[@]}"; do
  CMD+=("--override" "${ov}")
done

echo "Submitting training job: ${JOB_NAME}"
printf "Command: %q " "${CMD[@]}"; echo

# Run job in the foreground. Use nohup if you want it detached.
"${CMD[@]}"

# Sync artifacts and runs back to GCS
if [[ -n "${GCS_BUCKET}" ]]; then
  echo "Syncing artifacts -> ${GCS_BUCKET}/artifacts"
  gsutil -m rsync -r "${ARTIFACTS_DIR}" "${GCS_BUCKET}/artifacts"

  echo "Syncing runs -> ${GCS_BUCKET}/runs"
  gsutil -m rsync -r "${RUNS_DIR}" "${GCS_BUCKET}/runs"
fi
