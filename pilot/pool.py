"""Build the full candidate pool for the scaled run.

Geometry filters only; colour labelling happens afterwards in recolor.py so a
colour can be dropped without losing its neighbours.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .constants import CLASSES
from .data import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=float, default=44)
    ap.add_argument("--min-area", type=float, default=0.015)
    ap.add_argument("--max-area", type=float, default=0.35)
    ap.add_argument("--fill", type=float, default=0.45,
                    help="min mask/bbox area ratio, drops heavily occluded objects")
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--out", default=str(DATA / "pool.jsonl"))
    args = ap.parse_args()

    coco = json.load(open(DATA / "raw" / "annotations" / "instances_val2017.json"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    keep = CLASSES if not args.all_classes else set(cat_name.values())

    by = {}
    for ann in coco["annotations"]:
        if ann["iscrowd"] or not isinstance(ann["segmentation"], list):
            continue
        by.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    rows = []
    for (iid, cid), anns in by.items():
        name = cat_name[cid]
        if name not in keep or len(anns) != 1:
            continue
        ann = anns[0]
        im = images[iid]
        x, y, w, h = ann["bbox"]
        if w < args.min_side or h < args.min_side:
            continue
        ratio = (w * h) / (im["width"] * im["height"])
        if not (args.min_area <= ratio <= args.max_area):
            continue
        if ann["area"] / max(1.0, w * h) < args.fill:
            continue
        rows.append({
            "sample_id": f"{iid}_{cid}",
            "image_id": iid,
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
            "category": name,
            "bbox": [round(v, 1) for v in ann["bbox"]],
            "segmentation": ann["segmentation"],
            "area_ratio": round(ratio, 4),
            "fill": round(ann["area"] / max(1.0, w * h), 3),
        })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"pool {len(rows)} -> {args.out}")
    print("classes", len(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
