#!/usr/bin/env bash
set -euo pipefail

hs-tasnet eval --cfg src/hs_tasnet/config/eval.yaml --checkpoint runs/<run_id>/checkpoint_epoch1.pt
