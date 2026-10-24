#!/usr/bin/env bash
# VQA-v2 val questions + annotations (used only for the external colour task).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/vqa2"
mkdir -p "$DATA"
cd "$DATA"

[ -f v2_Questions_Val_mscoco.zip ] || \
  curl -L -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
[ -f v2_Annotations_Val_mscoco.zip ] || \
  curl -L -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

unzip -qo v2_Questions_Val_mscoco.zip
unzip -qo v2_Annotations_Val_mscoco.zip
ls -1 "$DATA"
