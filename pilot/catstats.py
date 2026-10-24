"""Colour statistics per COCO category, used to make absence hints plausible.

An absent-object hint should say something a knowledgeable annotator might say
("the umbrella is black"), not an impossible colour. We estimate each
category's colour distribution from its val instances.
"""

import argparse
import json
from collections import Counter, defaultdict

from PIL import Image

from .data import DATA
from .recolor import dominant_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-instances", type=int, default=25)
    ap.add_argument("--cap", type=int, default=200, help="instances per category")
    ap.add_argument("--out", default=str(DATA / "cat_colors.json"))
    args = ap.parse_args()

    coco = json.load(open(DATA / "raw" / "annotations" / "instances_val2017.json"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    by_cat = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["iscrowd"] or not isinstance(ann["segmentation"], list):
            continue
        by_cat[cat_name[ann["category_id"]]].append(ann)

    stats, cache = {}, {}
    for name, anns in sorted(by_cat.items()):
        if len(anns) < args.min_instances:
            continue
        counts = Counter()
        for ann in anns[:args.cap]:
            im = images[ann["image_id"]]
            key = im["id"]
            if key not in cache:
                cache[key] = Image.open(DATA / "images" / im["file_name"]).convert("RGB")
            color, _ = dominant_color(cache[key], ann["segmentation"])
            if color:
                counts[color] += 1
        total = sum(counts.values())
        if total >= args.min_instances:
            stats[name] = {"n": total,
                           "counts": dict(counts.most_common()),
                           "top": [c for c, _ in counts.most_common(3)]}
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"{len(stats)} categories -> {args.out}")
    for name in list(stats)[:8]:
        print(" ", name, stats[name]["counts"])


if __name__ == "__main__":
    main()
