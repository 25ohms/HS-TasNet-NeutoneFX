# Agents.md — HS-TasNet Project Agent (ML/AI Engineering)

This repository uses an AI agent to help set up and maintain a clean, reproducible, and production-ready ML codebase for training and inference of **HS-TasNet** (Hybrid Spectrogram-TasNet). The agent’s job is to establish and enforce best practices similar to high-quality ML repos (e.g., Hugging Face–style ergonomics: clear configs, reproducible runs, robust CLI tools, and well-scoped modules).

## Repository context (current state)

- `src/`
  - currently contains: `export_hs_tasnet.py` (exports a `.nm` file from a pretrained checkpoint)
- `artifacts/`
  - stores checkpoints (weights). Treat it as a cache/artifact directory, not a source directory.
- `README.md`
  - explains the purpose of why the author is making this repo (anything regarding documentation and setup, please add it below what already exists)

The agent may reorganize the repo while keeping compatibility with the current export flow.

---

## Agent mission

You are responsible for:
1. Designing and implementing a **clean project skeleton** for HS-TasNet training + inference.
2. Setting up **reproducible tooling**: environment, configs, logging, checkpoints, tests.
3. Writing code that is **portable across environments**:
   - local CPU
   - CUDA GPU (single and multi-GPU)
   - limited hardware (laptop/CI)
4. Providing **simple CLI entry points** for:
   - training
   - evaluation
   - inference / streaming inference
   - export (existing `export_hs_tasnet.py`)
5. Following best ML engineering practices:
   - configuration-driven pipelines
   - deterministic/reproducible runs
   - clean separation of concerns
   - clear documentation

---

## Target project structure

The agent should converge towards a layout like:

```

.
├── Agents.md
├── pyproject.toml               # preferred (or requirements.txt)
├── README.md
├── src/
│   ├── hs_tasnet/
│   │   ├── **init**.py
│   │   ├── config/
│   │   │   ├── defaults.yaml
│   │   │   ├── train.yaml
│   │   │   ├── eval.yaml
│   │   │   ├── infer.yaml
│   │   │   └── export.yaml
│   │   ├── data/
│   │   │   ├── datasets.py       # MusDB etc (pluggable)
│   │   │   ├── transforms.py     # augmentations
│   │   │   └── collate.py
│   │   ├── models/
│   │   │   ├── hs_tasnet.py      # architecture
│   │   │   ├── modules.py        # encoder/decoder, LSTM blocks, mask heads
│   │   │   └── streaming.py      # stateful streaming wrapper
│   │   ├── losses/
│   │   │   ├── waveform.py       # L1/L2, SI-SNR variants if used
│   │   │   └── spectral.py       # optional multi-domain losses
│   │   ├── train/
│   │   │   ├── train.py          # main training loop
│   │   │   ├── evaluate.py       # validation + metrics
│   │   │   ├── optim.py          # schedulers, optimizer builders
│   │   │   └── checkpointing.py  # save/load/resume
│   │   ├── infer/
│   │   │   ├── infer.py          # offline inference on files/batches
│   │   │   └── realtime.py       # low-latency streaming inference
│   │   ├── export/
│   │   │   └── export_hs_tasnet.py # existing script, may be relocated but keep entrypoint
│   │   ├── utils/
│   │   │   ├── logging.py        # structured logs, tqdm, wandb optional
│   │   │   ├── seed.py           # seeding/determinism
│   │   │   ├── audio.py          # STFT helpers, resampling, I/O
│   │   │   ├── device.py         # device selection
│   │   │   └── typing.py
│   │   └── cli/
│   │       ├── main.py           # `python -m hs_tasnet ...`
│   │       ├── train_cli.py
│   │       ├── eval_cli.py
│   │       ├── infer_cli.py
│   │       └── export_cli.py
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
├── tests/
│   ├── test_shapes.py            # quick forward shape checks
│   ├── test_streaming.py         # stateful inference behavior
│   └── test_export.py
├── artifacts/                       # checkpoints only (gitignored or selective)
├── runs/                         # logs, checkpoints, artifacts (gitignored)
└── .github/workflows/ci.yml

```

Guidelines:
- `artifacts/` remains the checkpoint storage directory (do not commit large files).
- `runs/` contains experiment outputs (logs, saved configs, checkpoints).
- Keep the current `src/export_hs_tasnet.py` working; a wrapper entrypoint is acceptable.

---

## Non-negotiable best practices

### Reproducibility
- Use a `set_seed(seed)` helper that seeds:
  - Python `random`
  - NumPy
  - PyTorch CPU and CUDA
- Log all seeds and config values.
- Save a copy of the exact config used for every run into `runs/<run_id>/config.yaml`.

### Configuration-driven execution
- Training/eval/infer/export must be driven by YAML config files (Hydra-like style is fine, but keep it lightweight).
- Agent must support overriding config from CLI (e.g., `--cfg config/train.yaml --override train.batch_size=8`).

### Clear device + precision support
Support:
- CPU
- CUDA
- mixed precision (`torch.cuda.amp.autocast`) when enabled
- gradient accumulation when batch size is limited

### Minimal, robust CLI
Provide:
- `hs-tasnet train --cfg ...`
- `hs-tasnet eval --cfg ...`
- `hs-tasnet infer --cfg ...`
- `hs-tasnet export --cfg ...`

Commands must:
- print the resolved config
- validate paths and shapes early
- fail with helpful errors

### Checkpointing and resuming
- Save:
  - model state
  - optimizer state
  - scheduler state
  - step/epoch counters
  - RNG state if possible
- Allow `--resume` from checkpoint.

### Logging
- Use structured logs (timestamp, level, step, metrics).
- Prefer `tensorboard` by default.
- Optional: Weights & Biases if a config flag is enabled (never mandatory).

### Tests
At minimum:
- forward pass shape test for a small batch
- streaming inference step consistency test (state handling)
- export script smoke test (can be skipped if `.nm` toolchain unavailable in CI)

### Code quality
- Type hints (`typing`)
- Format/lint (ruff or black)
- Avoid tight coupling between training and model code.
- Keep functions small and testable.

---

## HS-TasNet-specific engineering requirements

### Model modules (expected)
Implement these as reusable modules:
- `SpectrogramEncoder` (STFT wrapper)
- `ConvEncoder` (Conv1d learned filterbank)
- `MemoryLSTMBlock` (LSTM + skip + optional projection)
- `Fusion` (concat or sum, matching HS-TasNet vs HS-TasNet-Small)
- `MaskHead` (predict 4 masks per domain)
- `SpectrogramDecoder` (iSTFT wrapper)
- `ConvDecoder` (ConvTranspose1d learned synthesis filterbank)
- `HybridCombiner` (sum outputs per stem)

### Streaming inference
Provide an explicit stateful wrapper:
- caches:
  - LSTM hidden states for each memory block
  - necessary overlap buffers (e.g., for iSTFT overlap-add)
- API example:
  - `y = model.step(x_hop)` where `x_hop` is hop-sized input
- Must be deterministic and stable over long runs.

### Latency and framing
- Respect the paper’s framing choices:
  - window=1024, hop=512 at 44.1kHz (default configs)
- Provide config knobs for window/hop to experiment safely.

### Export compatibility
- Keep the ability to export from a pretrained checkpoint into `.nm`.
- The agent may refactor `export_hs_tasnet.py`, but must preserve:
  - existing CLI behavior (or provide a compatibility wrapper)
  - ability to load checkpoints from `artifacts/`

---

## Tooling requirements

### Environment
Prefer `pyproject.toml` (poetry) or `requirements.txt`.
Include:
- torch
- torchaudio (if used for I/O)
- numpy
- tqdm
- pyyaml
- soundfile
- tensorboard
Optional:
- wandb
- hydra-core (only if justified; keep config system simple)

### CI
Add a GitHub Actions workflow that:
- installs dependencies
- runs formatting/lint
- runs unit tests on CPU with tiny inputs

### Data handling
- Data loaders must be memory-safe and streaming-friendly.
- Support:
  - MusDB-like dataset structure
  - optional user dataset directory
- Provide a “tiny synthetic dataset” mode for CI.

---

## Performance and portability policies

The agent must write code that:
- avoids unnecessary copies
- uses `torch.no_grad()` for inference
- uses AMP for GPU when enabled
- avoids hardcoding CUDA
- supports setting `num_workers`, `pin_memory`, `prefetch_factor` from config

Also:
- provide a CPU-safe fallback for everything
- keep batch sizes small in examples
- ensure code runs on macOS (MPS optional but not required)

---

## Security / safety and artifact handling

- Do not commit large checkpoints.
- Add `artifacts/` and `runs/` to `.gitignore` by default, with guidance on how to store artifacts.
- If a checkpoint download feature is added, it must:
  - verify hashes when possible
  - store to `artifacts/` or `~/.cache/hs_tasnet/`

---

## Agent “definition of done” checklist

A change is complete when:
- `hs-tasnet train` can run on a tiny dataset and save checkpoints
- `hs-tasnet eval` loads a checkpoint and reports metrics
- `hs-tasnet infer` produces 4 stems for an input file
- streaming wrapper can process hop-by-hop without crashing
- export script works with checkpoints in `artifacts/`
- tests pass in CI
- configs and docs exist for typical workflows

---

## Suggested initial tasks for the agent

1. Create package structure under `src/hs_tasnet/`
2. Implement minimal HS-TasNet model skeleton with correct shapes
3. Add config system (YAML + CLI overrides)
4. Add training loop (single-GPU and CPU)
5. Add inference CLI (offline file separation)
6. Add streaming wrapper API
7. Refactor/bridge `export_hs_tasnet.py` into `src/hs_tasnet/export/`
8. Add tests and CI workflow
9. Update README with quickstart commands

---

## Notes to the agent

- Do not over-engineer. Start with a working baseline, then iterate.
- Keep the repo usable by a single developer on a laptop.
- Prefer clarity and correctness over micro-optimizations.
- Preserve backwards compatibility for existing export usage.
