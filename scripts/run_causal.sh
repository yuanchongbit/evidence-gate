#!/usr/bin/env bash
# Causal experiments on Qwen2.5-VL-3B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

python -m pilot.steer      --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --run runs/v5_261.jsonl --out runs/steer_261.jsonl
python -m pilot.steer_analyze --run runs/steer_261.jsonl
python -m pilot.steer_half --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --run runs/v5_261.jsonl --out runs/steer_half.jsonl
python -m pilot.steer_analyze --run runs/steer_half.jsonl

python -m pilot.patch3 --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --layers 24,32 --out runs/patch3.jsonl
python -m pilot.patch_analyze --out runs/patch3.jsonl
