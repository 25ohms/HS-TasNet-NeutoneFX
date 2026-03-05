# Vertex AI Submission Requirements

Provide the following before submitting a training job:

1. GCP project
- `PROJECT_ID`
- `REGION` (e.g. `us-central1`)

2. GCS buckets
- `STAGING_BUCKET` for Vertex SDK staging (e.g. `gs://your-bucket/staging`)
- `DATASET_URI` for MusDB18 (e.g. `gs://your-bucket/musdb18`)
- `BASE_OUTPUT_DIR` for model artifacts (e.g. `gs://your-bucket/model-artifacts`)
- Optional: `GCS_RUNS_URI` if you want run logs copied back (e.g. `gs://your-bucket/runs`)

3. Container image
- `CONTAINER_URI` pointing to Artifact Registry
- The image must include `scripts/vertex_worker.py`

4. Model serving container
- `MODEL_SERVING_CONTAINER` (e.g. `us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest`)

5. Auth & permissions
- Orchestrator has Application Default Credentials
- Service account has roles:
  - `aiplatform.admin`
  - `storage.objectAdmin` (or narrower read/write on your buckets)

6. Optional job settings
- `MACHINE_TYPE` (e.g. `g2-standard-4` for L4)
- `ACCELERATOR_TYPE` and `ACCELERATOR_COUNT`
- `REPLICA_COUNT`
- `SERVICE_ACCOUNT` for the training job (if different)
- `NETWORK` if you run in a VPC

7. Config overrides (optional)
- Any CLI overrides for training (e.g. `train.batch_size=4`)

Example submission:

```bash
python scripts/vertex_orchestrator.py \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --staging-bucket "$STAGING_BUCKET" \
  --container-uri "$CONTAINER_URI" \
  --model-serving-container "$MODEL_SERVING_CONTAINER" \
  --base-output-dir "$BASE_OUTPUT_DIR" \
  --dataset-uri "$DATASET_URI"
```
