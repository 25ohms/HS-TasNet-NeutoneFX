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

## Artifacts

- `artifacts/` stores checkpoints (do not commit large files).
- `runs/` stores logs, configs, and checkpoints for each run.
