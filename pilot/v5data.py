"""Build the scaled v5 manifest: verified colour labels + verified-absent category.

The v5 design needs no image editing: the "insufficient evidence" arm asks
about a category that COCO annotations place outside the scene, and the hint
gives that category one of its usual colours.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from PIL import Image

from .constants import DISTRACTOR
from .data import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "pool.jsonl"))
    ap.add_argument("--labels", default=str(DATA / "labels_verified.json"))
    ap.add_argument("--cat-colors", default=str(DATA / "cat_colors.json"))
    ap.add_argument("--out", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--img-dir", default=str(DATA / "v5"))
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = [json.loads(l) for l in open(args.pool)]
    labels = json.load(open(args.labels))
    stats = json.load(open(args.cat_colors))
    coco = json.load(open(DATA / "raw" / "annotations" / "instances_val2017.json"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    present = Counter()
    for ann in coco["annotations"]:
        present[(ann["image_id"], cat_name[ann["category_id"]])] += 1

    img_dir = DATA / "v5"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_rows, skipped = [], 0
    for r in pool:
        sid = r["sample_id"]
        if sid not in labels:
            continue
        gold = labels[sid]
        banned = {c for (iid, c), n in present.items() if iid == r["image_id"]}
        options = [c for c in stats if c not in banned and c != r["category"]]
        if not options:
            skipped += 1
            continue
        acat = rng.choice(options)
        top = stats[acat]["top"][:3]
        acolor = rng.choice(top)
        alt = next((c for c in top if c != acolor), None)
        if alt is None:
            alt = next((c for c in stats[acat]["counts"] if c != acolor), None)
        if alt is None:
            skipped += 1
            continue

        src = DATA / "images" / r["file_name"]
        img = Image.open(src).convert("RGB")
        if max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        dst = img_dir / f"{sid}.jpg"
        img.save(dst, quality=92)
        out_rows.append({
            "sample_id": sid,
            "image_id": r["image_id"],
            "file_name": r["file_name"],
            "category": r["category"],
            "bbox": r["bbox"],
            "gold": gold,
            "distractor": DISTRACTOR[gold],
            "absent_category": acat,
            "absent_hint": acolor,
            "absent_hint_alt": alt,
            "verified": True,
            "renders": {"orig": str(dst.relative_to(DATA.parent))},
        })

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(out_rows)} rows -> {args.out}  (skipped {skipped})")
    print("colours", Counter(r["gold"] for r in out_rows))
    print("absent cats", len({r["absent_category"] for r in out_rows}))


if __name__ == "__main__":
    main()
