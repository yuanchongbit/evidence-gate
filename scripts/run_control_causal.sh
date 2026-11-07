#!/usr/bin/env bash
# Causal controls for the evidence-direction result.
#
# 3B (default):
#   bash scripts/run_control_causal.sh
# 7B:
#   TAG=7b MODEL="$PWD/models/qwen7b" RUN=runs/v5_261_7b.jsonl \
#     LAYERS=8,14,20,26 PATCH_LAYERS=24,26 bash scripts/run_control_causal.sh
# LLaVA:
#   TAG=llava MODEL="$PWD/models/llava_ov_7b" FAMILY=gen RESIZE=512 \
#     RUN=runs/v5_261_llava.jsonl LAYERS=8,14,20,26 PATCH_LAYERS=24,26 \
#     bash scripts/run_control_causal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
FAMILY="${FAMILY:-qwen}"
RESIZE="${RESIZE:-512}"
RUN="${RUN:-runs/v5_261.jsonl}"
TAG="${TAG:-3b}"
LAYERS="${LAYERS:-8,16,24,32}"
PATCH_LAYERS="${PATCH_LAYERS:-24,32}"

python -m pilot.control_steer --manifest data/manifest_v5.jsonl \
  --model "$MODEL" --family "$FAMILY" --resize "$RESIZE" --run "$RUN" \
  --layers "$LAYERS" --out "runs/control_steer_${TAG}.jsonl"
python -m pilot.patch5 --manifest data/manifest_v5.jsonl \
  --model "$MODEL" --family "$FAMILY" --resize "$RESIZE" \
  --layers "$PATCH_LAYERS" --out "runs/patch5_${TAG}.jsonl"
python -m pilot.control_analyze --steer "runs/control_steer_${TAG}.jsonl" \
  --patch "runs/patch5_${TAG}.jsonl" --out "results/control_causal_${TAG}.json"
