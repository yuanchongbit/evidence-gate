#!/usr/bin/env bash
# Fourth backbone: a non-Qwen family with an abstention behaviour.
# Gemma-3-4B-it (Gemma 3 LLM + SigLIP) is loaded through the generic
# AutoModelForImageTextToText path in pilot/infer_gen.py. Use bf16: the
# bf16-trained weights overflow to NaN under fp16, which infer_gen now rejects.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-$ROOT/models/gemma3_4b}"
DTYPE="${DTYPE:-bfloat16}"
RUN="${RUN:-runs/v5_261_gemma.jsonl}"

python -m pilot.infer_gen --manifest data/manifest_v5.jsonl \
  --model "$MODEL" --out "$RUN" --resize 512 --dtype "$DTYPE"
python -m pilot.analyze_v5 --run "$RUN"
python -m pilot.probe --run "$RUN" --pairs none,hint,correct
python -m pilot.selective --run "$RUN"
python -m pilot.geometry --run "$RUN" --out results/geometry_gemma.json
