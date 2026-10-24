#!/usr/bin/env bash
# Same 261 samples on two more checkpoints, including a second model family.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODELS="${MODELS:-$ROOT/models}"

# Qwen2.5-VL-7B
python -m pilot.infer_gen --manifest data/manifest_v5.jsonl \
  --model "$MODELS/qwen25vl7b" --out runs/v5_261_7b.jsonl
python -m pilot.analyze_v5  --run runs/v5_261_7b.jsonl
python -m pilot.probe      --run runs/v5_261_7b.jsonl --pairs none,hint,correct
python -m pilot.selective  --run runs/v5_261_7b.jsonl
python -m pilot.patch3 --manifest data/manifest_v5.jsonl --model "$MODELS/qwen25vl7b" \
  --layers 18,24,26 --out runs/patch3_7b.jsonl
python -m pilot.patch_analyze --out runs/patch3_7b.jsonl

# LLaVA-OneVision-7B (different family, loaded through AutoModelForImageTextToText)
python -m pilot.infer_gen --manifest data/manifest_v5.jsonl \
  --model "$MODELS/llava_ov_7b" --out runs/v5_261_llava.jsonl --resize 768
python -m pilot.analyze_v5  --run runs/v5_261_llava.jsonl
python -m pilot.probe      --run runs/v5_261_llava.jsonl --pairs none,hint,correct
python -m pilot.selective  --run runs/v5_261_llava.jsonl
python -m pilot.patch4 --manifest data/manifest_v5.jsonl --model "$MODELS/llava_ov_7b" \
  --layers 18,24,26 --out runs/patch4_llava.jsonl
python -m pilot.patch_analyze --out runs/patch4_llava.jsonl
