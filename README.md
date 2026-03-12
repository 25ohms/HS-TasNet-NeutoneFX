# What is this repo?

- This repo contains [HS-TasNet](https://arxiv.org/pdf/2402.17701) checkpoints for Audio Source separation for Drums, Bass, Vocals, Other.
- I am currently working on writing my own implementation of [HS-TasNet](https://arxiv.org/pdf/2402.17701) based off of the paper

# Why does this exist?

- My goal is to improve Music Stem separation capabilities so that it can be used in realtime too. The importance of this is that with low-latency realtime methods, live audio can be demixed and provided as information for whatever purpose. My specific purpose is to improve light design by having stronger understandings of drum features, vocal features, and melodic features. See the [livingCAGE](https://github.com/25ohms/livingCAGE) for more information regarding my work on creating my own light design.

# Why Neutone FX?

- Neutone FX allows the option to import models with the .nm file format, which is great because it already works as a VST. My goal is to import my best model checkpoint I get onto Neutone FX and then import onto TouchDesigner for realtime stem-sepration feeding my audiovisual networks.

# Setup and usage

## Install (do not run yet)

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Notes:
- The `dev` extra installs `pytest` and `ruff` for tests/linting.
- Exporting to Neutone `.nm` requires `neutone_sdk` (install separately only if you need export).

## Quickstart

Training on the tiny synthetic dataset:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train.yaml
```

Evaluation:

```bash
hs-tasnet eval --cfg src/hs_tasnet/config/eval.yaml --checkpoint runs/<run_id>/checkpoint_epoch1.pt
```

Offline inference:

```bash
hs-tasnet infer --cfg src/hs_tasnet/config/infer.yaml \
  --override infer.input_path=path/to/audio.wav \
  --override infer.checkpoint=artifacts/hs_tasnet.pt
```

Export (Neutone):

```bash
hs-tasnet export --cfg src/hs_tasnet/config/export.yaml
```

## Configuration overrides

Override any YAML value from the CLI:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train.yaml --override train.batch_size=8
```

Example model-width overrides:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train.yaml \
  --override model.wave_lstm_hidden=500 \
  --override model.spec_lstm_hidden=500 \
  --override model.shared_lstm_hidden=1000
```

Current spectral mask parameterization is magnitude-only:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train.yaml \
  --override model.spec_mask_representation=magnitude
```

## Artifacts

- `artifacts/` stores checkpoints (do not commit large files).
- `runs/` stores logs, configs, and checkpoints for each run.

## GCP VM Orchestration

The repo includes scripts to bootstrap a VM and submit training jobs using a GCS bucket.

Example:

```bash
export GCS_BUCKET=gs://your-bucket
scripts/gcp_submit_train.sh
```

Common overrides:

```bash
export EXTRA_OVERRIDES=$'train.batch_size=4\ntrain.epochs=5'
export SYNC_MODE=rsync  # or mount if gcsfuse is installed
scripts/gcp_submit_train.sh
```

Config preset for VM paths:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/gcp.yaml
```

## Vertex AI Custom Jobs

Submit a training job from an orchestrator VM using Vertex AI (SDK-based orchestration):

```bash
python -m hs_tasnet.vertex_orchestrator \
  --project your-project \
  --region us-central1 \
  --staging-bucket gs://your-bucket/staging \
  --container-uri us-central1-docker.pkg.dev/your-project/your-repo/hs-tasnet:latest \
  --model-serving-container us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest \
  --base-output-dir gs://your-bucket/model-artifacts \
  --dataset-uri gs://your-bucket/musdb18
```

The worker container downloads the dataset from GCS, runs training, and writes the final checkpoint to `AIP_MODEL_DIR` so Vertex AI can register it in the Model Registry.

Build and push the training image to Artifact Registry:

```bash
export PROJECT_ID=your-project
export REGION=us-central1
export REPO=your-repo
export IMAGE_NAME=hs-tasnet
export TAG=latest
scripts/vertex_build_push.sh
```

Optional: choose a CUDA base image to reduce repeated pushes:

```bash
export BASE_IMAGE=pytorch/pytorch:latest
scripts/vertex_build_push.sh
```

Manual worker pool spec template (for `gcloud ai custom-jobs create`):

```
scripts/vertex_worker_pool.json
```

For MUSDB (.stem.mp4) training, the container must include `musdb`, `stempeg`, and `ffmpeg`.

## Vertex AI Evaluation Jobs

Submit a separate evaluation job for a checkpoint stored in GCS and attach the result to a registered Vertex model:

```bash
export PROJECT_ID=your-project
export REGION=us-central1
export STAGING_BUCKET=gs://your-bucket/staging
export CONTAINER_URI=us-central1-docker.pkg.dev/your-project/your-repo/hs-tasnet:latest
export MODEL_URI=gs://realtime-stems-model-artifacts/model
export DATASET_URI=gs://your-bucket/musdb18
export MODEL_RESOURCE_NAME=projects/123456789/locations/us-central1/models/987654321
export EVAL_OUTPUT_URI=gs://your-bucket/evaluations/hs-tasnet-musdb18-test
export SERVICE_ACCOUNT=vertex-jobs@your-project.iam.gserviceaccount.com

python -m hs_tasnet.vertex_eval_orchestrator
```

Mandatory environment variables for evaluation submission:
- `PROJECT_ID`
- `REGION`
- `STAGING_BUCKET`
- `CONTAINER_URI`
- `MODEL_URI`
- `DATASET_URI`
- `MODEL_RESOURCE_NAME`
- `EVAL_OUTPUT_URI`
- `SERVICE_ACCOUNT`

The eval worker downloads `model.pt` and `config.yaml` from `MODEL_URI`, runs checkpoint evaluation against the supplied dataset, writes:
- `metrics.json`
- `row_metrics.jsonl`
- `model_registry_import.json`

and then imports the aggregate metrics onto the registered model in Vertex Model Registry.

## PR checker

This repo now includes a separate GitHub Actions workflow at `.github/workflows/pr-checker.yml` for pull request review.

What it does:
- runs on `pull_request_target` so it can safely use repository secrets without executing PR code
- fetches PR metadata and changed files through the GitHub API
- sends the reviewed diff to OpenAI and asks for findings focused on:
  - security and secret-handling issues
  - training breakage or silent regressions
  - inference and export compatibility issues
  - workflow/config problems that could affect CI or deployment
- posts or updates a sticky PR comment with the review summary
- fails the check if the model returns blocking findings

Required setup:

1. In your GitHub repository settings, create a repository secret named `OPENAI_API_KEY`.
2. Open pull requests normally. The existing `CI` workflow still handles lint/tests, and `PR Checker` adds the AI review layer.

Optional local install for dry-runs:

```bash
pip install -e ".[review]"
```

Model selection:
- The workflow defaults to `gpt-4.1-mini` via `OPENAI_MODEL`.
- If you want a different model, change the `OPENAI_MODEL` value in `.github/workflows/pr-checker.yml`.

Security note:
- `pull_request_target` is used intentionally here. The workflow checks out the base branch version of the repository and reads PR diffs from the GitHub API, which avoids running untrusted pull request code while the OpenAI secret is available.
