#!/usr/bin/env bash
# VizWiz val parquet (images are embedded in the parquet files).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/vizwiz_ms"
mkdir -p "$DEST"
modelscope download --dataset lmms-lab/VizWiz-VQA --local_dir "$DEST"
ls -1 "$DEST" "$DEST/data" 2>/dev/null | head
