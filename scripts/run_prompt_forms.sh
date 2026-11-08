#!/usr/bin/env bash
# Six prompt forms on the full absent manifest, Qwen2.5-VL 3B then 7B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/runs}"
mkdir -p "$OUT" "$ROOT/logs"
M3="${M3:-$ROOT/.cache/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master}"
M7="${M7:-$ROOT/models/qwen7b}"
G4="${G4:-$ROOT/models/gemma3_4b}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "3B start"
python -m pilot.prompt_forms --model "$M3" --out "$OUT/prompt_forms_3b.jsonl"
log "3B done"

log "7B start"
python -m pilot.prompt_forms --model "$M7" --out "$OUT/prompt_forms_7b.jsonl"
log "7B done"

log "analysis"
python -m pilot.prompt_forms_analyze "$OUT/prompt_forms_3b.jsonl" std
python -m pilot.prompt_forms_analyze "$OUT/prompt_forms_7b.jsonl" std
log "Gemma battery"
python -m pilot.prompt_forms_gen --model "$G4" --out "$OUT/prompt_forms_gemma.jsonl"
log "Gemma dialogue no-hint fix"
python -m pilot.pf_dialogue_fix "$OUT/pf_gemma_dialogue_fix.jsonl"
python -m pilot.pf_merge_fix "$OUT/prompt_forms_gemma.jsonl" \
    "$OUT/pf_gemma_dialogue_fix.jsonl" "$OUT/prompt_forms_gemma.npz" \
    "$OUT/pf_gemma_dialogue_fix.npz" "$OUT/prompt_forms_gemma_fixed"
python -m pilot.prompt_forms_analyze "$OUT/prompt_forms_gemma_fixed.jsonl" std
log "ALL DONE"
