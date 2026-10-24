"""Filesystem layout: everything generated lives under <repo>/data.

The small frozen inputs (manifest_v5.jsonl, labels_verified.json,
cat_colors.json, pool.jsonl) are versioned; images and annotations are
downloaded by scripts/download_coco.sh.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
