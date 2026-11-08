#!/usr/bin/env bash
# Energy-matched removal audit (extra44, --lite) for the two non-Qwen backbones.
set -u
ROOT=/mnt/workspace
cd "$ROOT"
GEMMA="$ROOT/models/gemma3_4b"
PHI="$ROOT/.cache/modelscope/models/models/LLM-Research--Phi-3.5-vision-instruct/snapshots/master"
[ -d "$PHI" ] || PHI="$ROOT/models/phi35_vision"
OUT="$ROOT/results/runs"
STATUS="$ROOT/results/extra44_nonqwen_status.log"
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

smoke () {
  name="$1"
  shift
  log="$OUT/${name}_smoke.log"
  echo "[$(date +%H:%M:%S)] start $name smoke" >> "$STATUS"
  if "$@" > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] $name smoke rc=0" >> "$STATUS"
  else
    echo "[$(date +%H:%M:%S)] $name smoke rc=$? (see $log)" >> "$STATUS"
    echo "[$(date +%H:%M:%S)] aborted after $name smoke failure" >> "$STATUS"
    exit 1
  fi
}

smoke gemma_extra44 python -m pilot.extra44 --lite --limit 2 --family gen --model "$GEMMA" \
  --run "$ROOT/results/runs/v5_261_gemma.jsonl" --out /tmp/extra44_gemma_smoke.jsonl \
  --stats /tmp/extra44_gemma_smoke.json

smoke phi_extra44 python -m pilot.extra44 --lite --limit 2 --family phi --model "$PHI" \
  --run "$ROOT/results/runs/phi_261.jsonl" --out /tmp/extra44_phi_smoke.jsonl \
  --stats /tmp/extra44_phi_smoke.json

step gemma_extra44 python -m pilot.extra44 --lite --family gen --model "$GEMMA" \
  --run "$ROOT/results/runs/v5_261_gemma.jsonl" --out "$OUT/extra44_gemma.jsonl" \
  --stats "$ROOT/results/extra44_gemma_stats.json"

step phi_extra44 python -m pilot.extra44 --lite --family phi --model "$PHI" \
  --run "$ROOT/results/runs/phi_261.jsonl" --out "$OUT/extra44_phi.jsonl" \
  --stats "$ROOT/results/extra44_phi_stats.json"

echo "[$(date +%H:%M:%S)] done" >> "$STATUS"
