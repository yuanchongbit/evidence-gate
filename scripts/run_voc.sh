#!/usr/bin/env bash
# Second stimulus set: PASCAL VOC 2012 segmentation, same six cells and 3B model.
# Requires VOCtrainval_11-May-2012 unpacked at VOC_ROOT (default data/VOCdevkit/VOC2012).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VOC_ROOT="${VOC_ROOT:-$ROOT/data/VOCdevkit/VOC2012}"
MANIFEST="${MANIFEST:-data/manifest_voc.jsonl}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN="${RUN:-runs/voc_3b.jsonl}"

python -m pilot.voc_build --voc-root "$VOC_ROOT" --out "$MANIFEST"
python -m pilot.infer --manifest "$MANIFEST" --model "$MODEL" \
  --out "$RUN" --skip-probes --save-hidden
python -m pilot.voc_analyze --run "$RUN" --out results/voc_results.json
python -m pilot.voc_extra --run "$RUN" --out results/voc_extra.json
