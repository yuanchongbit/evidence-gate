#!/usr/bin/env bash
# Controls, task-form boundaries, and external validation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

# Controls: gray image, presence manipulation check, category-matched pairs.
python -m pilot.blank --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --out runs/blank.jsonl
python -m pilot.check_presence --manifest data/manifest_v5.jsonl --model "$MODEL" \
  --out runs/presence_v5.jsonl
python -m pilot.matched --build
python -m pilot.matched --presence --model "$MODEL"
python -m pilot.matched --model "$MODEL" --conflict

# Task-form boundaries.
python -m pilot.task2 --model "$MODEL" --out runs/task2.jsonl
python -m pilot.task_analyze --run runs/task2.jsonl
python -m pilot.task3 --model "$MODEL" --out runs/task3.jsonl
python -m pilot.task_analyze --run runs/task3.jsonl

# External validation (requires scripts/download_vizwiz.sh and download_vqa2.sh).
python -m pilot.ext_vizwiz --n-each 250 --out runs/ext_vizwiz.jsonl
python -m pilot.ext_analyze --run runs/ext_vizwiz.jsonl --main-run runs/v5_261.jsonl
python -m pilot.ext_vqa_color --n 300 --out runs/ext_vqa_color.jsonl
python -m pilot.ext_vqa_analyze --run runs/ext_vqa_color.jsonl
