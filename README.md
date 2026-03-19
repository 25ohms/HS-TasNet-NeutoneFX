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

Training on the tiny synthetic dataset (lightweight baseline/dev):

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train.yaml
```

Paper-faithful training config:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train_paper.yaml
```

`train_paper.yaml` now mirrors the Demucs-style training recipe more closely:
- `train.epochs=360`
- `train.batch_size=64`
- `optim.optim=adam`
- `optim.lr=3e-4`
- `optim.beta1=0.9`
- `optim.beta2=0.999`
- `optim.loss=l1`
- `optim.weight_decay=0`

Training controls that are useful when SDR stalls or training becomes unstable:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train_paper.yaml \
  --override optim.loss=mse \
  --override optim.clip_grad=1.0 \
  --override model.bottleneck_group_norm_groups=4 \
  --override regularization.singular_value.enabled=true \
  --override regularization.singular_value.weight=1e-4 \
  --override regularization.singular_value.target_patterns='["conv_encoder","conv_decoder","split"]'
```

Those switches are intentionally opt-in. The default remains architecture-faithful, while giving you a direct way to compare:
- pure L1 vs MSE vs SI-SNR objective
- no norm vs bottleneck group norm
- no spectral regularization vs low-rank singular-value penalty
- unclipped vs clipped gradients
- full epoch vs `train.max_batches` capped debugging runs

For VM training, keep [train_paper.yaml](/Users/sahal/Desktop/ohms/code/ml-ai/HS-TasNet-NeutoneFX/src/hs_tasnet/config/train_paper.yaml) as the single ground-truth config and enable SVD and/or GN with CLI overrides.

Weights & Biases logging:

```bash
pip install -e ".[wandb]"
export WANDB_API_KEY=your_wandb_api_key
hs-tasnet train --cfg src/hs_tasnet/config/train_paper.yaml \
  --override logging.wandb.enabled=true \
  --override logging.wandb.project=hs-tasnet \
  --override logging.wandb.entity=your_entity
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

Realtime streaming benchmark (Python API):

```python
from hs_tasnet.infer.realtime import benchmark_streaming, build_streaming
from hs_tasnet.models.hs_tasnet import HSTasNet

model = HSTasNet()
streamer = build_streaming(model.eval())
metrics = benchmark_streaming(streamer, num_hops=200)
print(metrics)  # avg_hop_ms, rtf, algorithmic_latency_ms
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

Demucs-style optimizer overrides:

```bash
hs-tasnet train --cfg src/hs_tasnet/config/train_paper.yaml \
  --override optim.optim=adam \
  --override optim.lr=3e-4 \
  --override optim.beta1=0.9 \
  --override optim.beta2=0.999 \
  --override optim.loss=l1
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

Config intent:
- `src/hs_tasnet/config/train.yaml` is the lightweight baseline/dev config.
- `src/hs_tasnet/config/train_paper.yaml` is the paper-faithful training config for real runs.
- `src/hs_tasnet/config/paper_hs_tasnet.yaml` contains paper-faithful model architecture defaults.

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

`gcp.yaml` inherits paper-faithful model settings and is intended for real dataset training on VM.

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

By default the orchestrator config (`config.yaml`) points training to:
- `src/hs_tasnet/config/train_paper.yaml`

`train_paper.yaml` uses `data.loader=musdb` and expects MUSDB at `/mnt/data/musdb18` inside the worker container (the Vertex worker populates this from `--dataset-uri`).
By default MUSDB validation uses the training subset split (`data.musdb_val_subset=train`, `data.musdb_val_split=valid`); reserve `test` for final testing/evaluation jobs.
If your MUSDB install does not expose a built-in `train/valid` split, the loader falls back to a deterministic 80/20 split on the training subset (`data.musdb_train_fraction=0.8`, `data.musdb_split_seed=42`).

The worker container downloads the dataset from GCS, runs training, and writes the final checkpoint to `AIP_MODEL_DIR` so Vertex AI can register it in the Model Registry.
Model artifacts are always written to a timestamped bundle directory under `AIP_MODEL_DIR`, for example:
- `.../model/20260312_153012_vertex-run-1234/model.pt`
- `.../model/20260312_153012_vertex-run-1234/config.yaml`

To reduce worker disk usage during long Vertex runs, the paper training preset now saves checkpoints every 5 epochs and retains only the latest 2 checkpoint files locally. You can override this with `train.ckpt_every` and `train.max_checkpoints`.
If you pass `--gcs-runs-uri`, the Vertex worker now syncs the run directory to GCS every epoch by default; set `vertex.train.gcs_sync_every_epochs` in the orchestrator config to use a different interval.

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

To enable W&B in Vertex jobs, pass your API key and logging overrides from the orchestrator environment:

```bash
export WANDB_API_KEY=your_wandb_api_key
python -m hs_tasnet.vertex_orchestrator \
  --cfg src/hs_tasnet/config/train_paper.yaml \
  --base-output-dir gs://realtime-stems-model-artifacts \
  --override logging.wandb.enabled=true \
  --override logging.wandb.project=hs-tasnet \
  --override logging.wandb.entity=your_entity
```

`vertex_orchestrator` forwards `WANDB_API_KEY` from the orchestrator VM environment into the Vertex worker container when supported by the installed Vertex SDK.

## Vertex AI Evaluation Jobs

Submit a separate evaluation job for a checkpoint stored in GCS and write the results to a GCS evaluation directory:

```bash
export PROJECT_ID=your-project
export REGION=us-central1
export STAGING_BUCKET=gs://your-bucket/staging
export CONTAINER_URI=us-central1-docker.pkg.dev/your-project/your-repo/hs-tasnet:latest
export MODEL_URI=gs://realtime-stems-model-artifacts/model
export DATASET_URI=gs://your-bucket/musdb18
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
- `EVAL_OUTPUT_URI`
- `SERVICE_ACCOUNT`

The eval worker downloads `model.pt` and `config.yaml` from `MODEL_URI`, runs checkpoint evaluation against the supplied dataset, and writes:
If `MODEL_URI` contains timestamped bundles, the worker automatically picks the latest bundle directory.
- `metrics.json`
- `row_metrics.jsonl`
- `resolved_config.yaml`

The evaluation job does not import results into Vertex Model Registry; it only stores artifacts in your bucket.

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
