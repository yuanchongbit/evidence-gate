#!/usr/bin/env bash
# Main 2x3 run on Qwen2.5-VL-3B + all no-GPU analyses.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

python -m pilot.infer --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --out runs/v5_261.jsonl --skip-probes --save-hidden
python -m pilot.analyze_v5  --run runs/v5_261.jsonl
python -m pilot.probe      --run runs/v5_261.jsonl --pairs none,hint,correct
python -m pilot.sdt        --run runs/v5_261.jsonl --out results/sdt.json
python -m pilot.baselines  --run runs/v5_261.jsonl
python -m pilot.selective  --run runs/v5_261.jsonl
