#!/usr/bin/env bash
set -euo pipefail

# Build and push the training image to Artifact Registry.
# Requires: gcloud auth, Artifact Registry repo created.

ORCHESTRATOR_CFG=${ORCHESTRATOR_CFG:-"config.yaml"}
PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-""}
REPO=${REPO:-""}
IMAGE_NAME=${IMAGE_NAME:-"hs-tasnet"}
TAG=${TAG:-"latest"}
BASE_IMAGE=${BASE_IMAGE:-""}

if [[ -f "${ORCHESTRATOR_CFG}" ]]; then
  CONFIG_VALUES=$(
    ORCHESTRATOR_CFG="${ORCHESTRATOR_CFG}" python - <<'PY'
from pathlib import Path
import os
import yaml

cfg_path = Path(os.environ["ORCHESTRATOR_CFG"])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
vertex = cfg.get("vertex", {})
container_uri = vertex.get("container_uri", "")
project_id = vertex.get("project_id", "")
region = vertex.get("region", "")
repo = ""
image_name = ""
tag = ""

if container_uri and ".pkg.dev/" in container_uri:
    left, right = container_uri.split(".pkg.dev/", 1)
    if not region:
        region = left
    parts = right.split("/")
    if len(parts) >= 3:
        if not project_id:
            project_id = parts[0]
        repo = parts[1]
        image_ref = parts[2]
        if ":" in image_ref:
            image_name, tag = image_ref.rsplit(":", 1)
        else:
            image_name = image_ref

for value in [project_id, region, repo, image_name, tag]:
    print(value or "")
PY
  )

  mapfile -t CONFIG_LINES <<< "${CONFIG_VALUES}"
  CONFIG_PROJECT_ID=${CONFIG_LINES[0]:-}
  CONFIG_REGION=${CONFIG_LINES[1]:-}
  CONFIG_REPO=${CONFIG_LINES[2]:-}
  CONFIG_IMAGE_NAME=${CONFIG_LINES[3]:-}
  CONFIG_TAG=${CONFIG_LINES[4]:-}

  PROJECT_ID=${PROJECT_ID:-"${CONFIG_PROJECT_ID}"}
  REGION=${REGION:-"${CONFIG_REGION}"}
  REPO=${REPO:-"${CONFIG_REPO}"}
  IMAGE_NAME=${IMAGE_NAME:-"${CONFIG_IMAGE_NAME}"}
  TAG=${TAG:-"${CONFIG_TAG}"}
fi

if [[ -z "${PROJECT_ID}" || -z "${REGION}" || -z "${REPO}" ]]; then
  echo "PROJECT_ID, REGION, and REPO are required. Set them directly or via ${ORCHESTRATOR_CFG}." >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "Building image: ${IMAGE_URI}"
if [[ -n "${BASE_IMAGE}" ]]; then
  docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_URI}" .
else
  docker build -t "${IMAGE_URI}" .
fi

echo "Pushing image: ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "Done: ${IMAGE_URI}"
