#!/usr/bin/env bash
# Control battery that was missing for the two non-Qwen backbones and for VOC:
# layer patching, conflict steering, random/ablation/layer steering controls,
# category-matched pairs, prompt formats, and the VOC layer scan.
# Each step is independent; a failure is recorded and the queue continues.
set -u
ROOT=/mnt/workspace
cd "$ROOT"
GEMMA="$ROOT/models/gemma3_4b"
PHI="$ROOT/.cache/modelscope/models/models/LLM-Research--Phi-3.5-vision-instruct/snapshots/master"
[ -d "$PHI" ] || PHI="$ROOT/models/phi35_vision"
Q3B="$ROOT/.cache/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master"
OUT="$ROOT/results/runs"
STATUS="$ROOT/results/missing_controls_status.log"
mkdir -p "$OUT"
: > "$STATUS"

step () {
  name="$1"
  shift
  log="$OUT/$name.log"
  echo "[$(date +%H:%M:%S)] start $name" >> "$STATUS"
  if "$@" > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] $name rc=0" >> "$STATUS"
  else
    echo "[$(date +%H:%M:%S)] $name rc=$? (see $log)" >> "$STATUS"
  fi
}

step voc_patch python -m pilot.patch3 --manifest data/manifest_voc.jsonl \
  --model "$Q3B" --layers 16,24,30,34 --out "$ROOT/runs/patch_voc.jsonl"

step gemma_patch python -m pilot.patch_full --family gen --model "$GEMMA" \
  --layers 8,16,24,30 --same-layers 24,30 --noop-layers 24,30 \
  --out "$OUT/patch_full_gemma.jsonl"

step phi_patch python -m pilot.patch_full --family phi --model "$PHI" \
  --layers 14,22,28,30 --same-layers 22,30 --noop-layers 22,30 \
  --out "$OUT/patch_full_phi.jsonl"

step gemma_steer python -m pilot.steer_conflict --family gen --model "$GEMMA" \
  --run "$ROOT/runs/v5_261_gemma.jsonl" --out "$OUT/steer_conflict_gemma.jsonl"

step phi_steer python -m pilot.steer_conflict --family phi --model "$PHI" \
  --run "$OUT/phi_261.jsonl" --out "$OUT/steer_conflict_phi.jsonl"

step gemma_matched python -m pilot.matched_gen --manifest data/manifest_matched.jsonl \
  --model "$GEMMA" --out "$OUT/matched_gemma.jsonl"

step phi_matched python -m pilot.matched_phi --manifest data/manifest_matched.jsonl \
  --model "$PHI" --out "$OUT/matched_phi.jsonl"

step gemma_prompt python -m pilot.prompt_variants --family gen --model "$GEMMA" \
  --out "$OUT/prompt_variants_gemma.jsonl"

step phi_prompt python -m pilot.prompt_variants --family phi --model "$PHI" \
  --out "$OUT/prompt_variants_phi.jsonl"

step gemma_control_steer python -m pilot.control_steer --family gen --model "$GEMMA" \
  --run "$ROOT/runs/v5_261_gemma.jsonl" --layers 8,16,24,30 \
  --out "$OUT/control_steer_gemma.jsonl"

step phi_control_steer python -m pilot.control_steer --family phi --model "$PHI" \
  --run "$OUT/phi_261.jsonl" --layers 14,22,28,30 \
  --out "$OUT/control_steer_phi.jsonl"

echo "[$(date +%H:%M:%S)] done" >> "$STATUS"
