#!/usr/bin/env bash
# Snapshot the three models used in the paper. Needs a Hugging Face token only
# for gated repositories; all three here are public.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:-$ROOT/models}"
mkdir -p "$MODELS"
HF="${HF_CLI:-hf}"

$HF download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$MODELS/qwen25vl3b"
$HF download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$MODELS/qwen25vl7b"
$HF download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir "$MODELS/llava_ov_7b"

echo "models in $MODELS"
