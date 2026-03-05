#!/usr/bin/env bash
set -euo pipefail

# Submit a Vertex AI CustomJob using a container image.
# Requires gcloud auth and a container image in Artifact Registry.

PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-""}
IMAGE_URI=${IMAGE_URI:-""}
GCS_BUCKET=${GCS_BUCKET:-""}
DATASET_PREFIX=${DATASET_PREFIX:-"musdb18"}
JOB_NAME=${JOB_NAME:-"hs-tasnet-$(date +%Y%m%d-%H%M%S)"}
CFG=${CFG:-"src/hs_tasnet/config/train.yaml"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-""}
ACCELERATOR_COUNT=${ACCELERATOR_COUNT:-"1"}
BOOT_DISK_GB=${BOOT_DISK_GB:-"200"}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}

if [[ -z "${PROJECT_ID}" || -z "${REGION}" || -z "${IMAGE_URI}" || -z "${GCS_BUCKET}" ]]; then
  echo "PROJECT_ID, REGION, IMAGE_URI, and GCS_BUCKET are required." >&2
  exit 1
fi

DATASET_URI="${GCS_BUCKET}/${DATASET_PREFIX}"
LOCAL_DATA_DIR="/mnt/data/${DATASET_PREFIX}"
ARTIFACTS_DIR="/mnt/artifacts"
RUNS_DIR="/mnt/runs"

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

OVERRIDE_ARGS=""
for ov in "${OVERRIDES[@]}"; do
  OVERRIDE_ARGS+=" --override ${ov}"
done

TRAIN_CMD=$(cat <<CMD
mkdir -p "${LOCAL_DATA_DIR}" "${ARTIFACTS_DIR}" "${RUNS_DIR}" && \
python scripts/gcs_sync.py --gcs-uri "${DATASET_URI}" --dest "${LOCAL_DATA_DIR}" --direction download && \
hs-tasnet train --cfg "${CFG}" ${OVERRIDE_ARGS} && \
python scripts/gcs_sync.py --gcs-uri "${GCS_BUCKET}/artifacts" --dest "${ARTIFACTS_DIR}" --direction upload && \
python scripts/gcs_sync.py --gcs-uri "${GCS_BUCKET}/runs" --dest "${RUNS_DIR}" --direction upload
CMD
)

ACCEL_BLOCK=""
if [[ -n "${ACCELERATOR_TYPE}" ]]; then
  ACCEL_BLOCK=$(cat <<ACC
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
ACC
)
fi

JOB_SPEC=$(cat <<YAML
displayName: ${JOB_NAME}
jobSpec:
  workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
${ACCEL_BLOCK}      diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${BOOT_DISK_GB}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      command: ["bash", "-lc"]
      args:
      - |
          ${TRAIN_CMD}
YAML
)

JOB_FILE="/tmp/vertex_job_${JOB_NAME}.yaml"
echo "${JOB_SPEC}" > "${JOB_FILE}"

echo "Submitting Vertex AI job: ${JOB_NAME}"
gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --config="${JOB_FILE}"
