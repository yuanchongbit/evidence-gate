#!/usr/bin/env bash
# COCO val2017 images + instance annotations (about 1.1 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/raw" "$DATA/images"
cd "$DATA/raw"

[ -f annotations_trainval2017.zip ] || \
  curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
[ -f val2017.zip ] || \
  curl -L -O http://images.cocodataset.org/zips/val2017.zip

unzip -qo annotations_trainval2017.zip annotations/instances_val2017.json
unzip -qo val2017.zip
mv val2017/*.jpg "$DATA/images/"
rmdir val2017

echo "images: $(ls "$DATA/images" | wc -l)  annotations: $DATA/raw/annotations/instances_val2017.json"
