#!/usr/bin/env bash
# Everything, in dependency order. Expect several GPU-hours on one 24 GB card.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/build_v5.sh
bash scripts/run_main.sh
bash scripts/run_causal.sh
bash scripts/run_cross_model.sh
bash scripts/run_controls.sh
