#!/usr/bin/env bash
set -euo pipefail

# Build and push the training image to Artifact Registry.
# Requires: gcloud auth, Artifact Registry repo created.

PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-""}
REPO=${REPO:-""}
IMAGE_NAME=${IMAGE_NAME:-"hs-tasnet"}
TAG=${TAG:-"latest"}
BASE_IMAGE=${BASE_IMAGE:-""}

if [[ -z "${PROJECT_ID}" || -z "${REGION}" || -z "${REPO}" ]]; then
  echo "PROJECT_ID, REGION, and REPO are required." >&2
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
