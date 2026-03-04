#!/usr/bin/env bash
set -euo pipefail

hs-tasnet infer --cfg src/hs_tasnet/config/infer.yaml --override infer.input_path=path/to/audio.wav --override infer.checkpoint=models/hs_tasnet.pt
