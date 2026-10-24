#!/usr/bin/env bash
# Rebuild data/manifest_v5.jsonl and the resized images from COCO.
# data/labels_verified.json, data/cat_colors.json and data/pool.jsonl are
# frozen inputs shipped with the repo; the manual contact-sheet check is not
# repeated here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pilot.v5data
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open("data/manifest_v5.jsonl")]
print("rows", len(rows), "colours", Counter(r["gold"] for r in rows))
print("absent categories", len({r["absent_category"] for r in rows}))
PY
