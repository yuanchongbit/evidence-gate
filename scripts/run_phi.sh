#!/usr/bin/env bash
# Second backbone outside the Qwen family: Phi-3.5-vision-instruct
# (Phi-3-mini LLM + CLIP ViT-L/14-336). Loaded with eager attention because
# the checkpoint config pins flash-attention-2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-$ROOT/models/phi35_vision}"
RUN="${RUN:-runs/v5_261_phi.jsonl}"
RESIZE="${RESIZE:-512}"
NUM_CROPS="${NUM_CROPS:-}"

EXTRA=()
if [ -n "$NUM_CROPS" ]; then EXTRA+=(--num-crops "$NUM_CROPS"); fi

python -m pilot.phi_infer --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --out "$RUN" --resize "$RESIZE" "${EXTRA[@]}"
python -m pilot.voc_analyze --run "$RUN" --out results/phi_results.json
python -m pilot.voc_extra --run "$RUN" --out results/phi_extra.json
