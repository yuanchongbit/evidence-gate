"""Second stimulus set: PASCAL VOC 2012 segmentation, same construction as v5.

The point of this run is external validity of the construction, not of the
model: COCO and VOC differ in images, scene statistics, class vocabulary and
annotation protocol, so a replication here cannot be explained by a COCO
peculiarity.  Labels come from automatic gates only (no contact-sheet
verification); the presence probe is run on the result as an independent check.

Construction mirrors v5:
  * present arm: one connected component of the class, box short side >= 44 px,
    bbox area 1.5-35% of the image, mask/bbox fill >= 0.45, dominant colour
    passing the same purity and margin gates as pilot/recolor.py;
  * absent arm: a VOC class with no annotated pixel in the image, hinted with
    one of that class's most frequent colours in the same split.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

from .constants import COLORS, DISTRACTOR
from .data import DATA

# VOC2012 classes; index 0 is background in the segmentation masks.
VOC = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
SKIP = {"background", "person", "pottedplant"}  # clothing / ambiguous colour

ACHROMATIC_S = 0.18
HUE_BINS = [
    ("red", 0, 12), ("orange", 12, 40), ("yellow", 40, 68),
    ("green", 68, 165), ("blue", 165, 255), ("purple", 255, 300),
    ("pink", 300, 348), ("red", 348, 360),
]


def rgb_to_hsv_array(px):
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = px.max(1)
    mn = px.min(1)
    diff = mx - mn
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    nz = diff > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & (mx == b) & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / diff[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / diff[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / diff[bm]) + 240
    return h, s, mx


def color_from_mask(arr, mask):
    """Dominant colour of a binary mask; same gates as pilot/recolor.py."""
    if mask.sum() < 200:
        return None
    px = arr[mask]
    h, s, v = rgb_to_hsv_array(px)
    achr = s < ACHROMATIC_S
    a_frac = float(achr.mean())
    if a_frac >= 0.6:
        vmed = float(np.median(v[achr]))
        if vmed >= 0.72:
            return "white"
        if vmed >= 0.42:
            return "gray"
        if vmed <= 0.25:
            return "black"
        return None
    chrom = ~achr
    c_frac = float(chrom.mean())
    if c_frac < 0.4:
        return None
    hh = h[chrom]
    counts = Counter()
    for name, lo, hi in HUE_BINS:
        counts[name] += int(((hh >= lo) & (hh < hi)).sum())
    ordered = counts.most_common()
    color, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0
    purity = top / max(1, int(chrom.sum()))
    margin = (top - second) / max(1, int(chrom.sum()))
    if purity < 0.6 or margin < 0.2 or c_frac < 0.5:
        return None
    vmed = float(np.median(v[chrom]))
    if color in ("orange", "yellow", "red"):
        warm = chrom & ((h >= 10) & (h < 50))
        if warm.sum() >= 50 and float(np.median(v[warm])) < 0.45:
            color = "brown"
    return color if color in COLORS else None


def components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 200:
            continue
        ys, xs = np.where(m)
        out.append({
            "mask": m,
            "bbox": [float(xs.min()), float(ys.min()),
                     float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
            "area": int(m.sum()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--target", type=int, default=180)
    ap.add_argument("--per-color", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "manifest_voc.jsonl"))
    ap.add_argument("--img-dir", default=str(DATA / "voc"))
    args = ap.parse_args()

    root = args.voc_root
    ids = [l.strip() for l in open(f"{root}/ImageSets/Segmentation/{args.split}.txt")
           if l.strip()]
    stats = defaultdict(Counter)

    # pass 1: per-class colour statistics over the split
    for n, iid in enumerate(ids, 1):
        seg = np.array(Image.open(f"{root}/SegmentationClass/{iid}.png"))
        arr = np.asarray(Image.open(f"{root}/JPEGImages/{iid}.jpg").convert("RGB"),
                         dtype=np.float32) / 255.0
        for idx, name in enumerate(VOC):
            if name in SKIP:
                continue
            m = seg == idx
            if m.sum() < 200:
                continue
            best = max(components(m), key=lambda c: c["area"], default=None)
            if best is None:
                continue
            col = color_from_mask(arr, best["mask"])
            if col:
                stats[name][col] += 1
        if n % 200 == 0:
            print(f"stats {n}/{len(ids)}", flush=True)

    print("class colour stats:")
    for name in sorted(stats):
        print(f"  {name:14s}", stats[name].most_common(6), flush=True)

    # pass 2: candidate pool
    cands = []
    for iid in ids:
        seg = np.array(Image.open(f"{root}/SegmentationClass/{iid}.png"))
        img = Image.open(f"{root}/JPEGImages/{iid}.jpg").convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = seg.shape
        for idx, name in enumerate(VOC):
            if name in SKIP:
                continue
            m = seg == idx
            if m.sum() < 200:
                continue
            comps = components(m)
            if len(comps) != 1:
                continue
            c = comps[0]
            bx, by, bw, bh = c["bbox"]
            if bw < 44 or bh < 44:
                continue
            frac = (bw * bh) / (w * h)
            if not (0.015 <= frac <= 0.35):
                continue
            if c["area"] / (bw * bh) < 0.45:
                continue
            col = color_from_mask(arr, c["mask"])
            if not col:
                continue
            cands.append({"image_id": iid, "category": name, "gold": col,
                          "bbox": [round(v, 1) for v in c["bbox"]],
                          "frac": round(frac, 4),
                          "present": {VOC[j] for j in np.unique(seg) if j < len(VOC)}})
    print(f"candidates {len(cands)}")

    # sample, one candidate per image, capped per colour
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    used_img, per_col = set(), Counter()
    rows = []
    for c in cands:
        if len(rows) >= args.target:
            break
        if c["image_id"] in used_img or per_col[c["gold"]] >= args.per_color:
            continue
        absent_opts = [n for n in stats if n not in c["present"] and n != c["category"]]
        if not absent_opts:
            continue
        acat = rng.choice(absent_opts)
        top = stats[acat].most_common(3)
        acolor = rng.choice([t[0] for t in top])
        used_img.add(c["image_id"])
        per_col[c["gold"]] += 1
        sid = f"{c['image_id']}_{c['category']}"
        rows.append({
            "sample_id": sid,
            "image_id": c["image_id"],
            "file_name": f"{c['image_id']}.jpg",
            "category": c["category"],
            "bbox": c["bbox"],
            "gold": c["gold"],
            "distractor": DISTRACTOR[c["gold"]],
            "absent_category": acat,
            "absent_hint": acolor,
            "absent_hint_alt": DISTRACTOR[acolor],
            "verified": False,
            "renders": {"orig": f"data/voc/{sid}.jpg"},
        })

    img_dir = DATA / "voc"
    img_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        img = Image.open(f"{root}/JPEGImages/{r['file_name']}").convert("RGB")
        if max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        img.save(img_dir / f"{r['sample_id']}.jpg", quality=92)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} rows -> {args.out}")
    print("colours", Counter(r["gold"] for r in rows))
    print("present classes", Counter(r["category"] for r in rows).most_common())
    print("absent classes", len({r["absent_category"] for r in rows}))

    # contact sheet for a spot check
    sheet = Image.new("RGB", (5 * 220, 8 * 190), "white")
    for i, r in enumerate(rows[:40]):
        im = Image.open(img_dir / f"{r['sample_id']}.jpg")
        im.thumbnail((210, 160))
        x, y = (i % 5) * 220 + 5, (i // 5) * 190 + 5
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.text((x, y + 162), f"{r['category']} / {r['gold']}", fill="black")
    sheet.save(DATA / "voc_sheet.jpg")
    print("sheet -> data/voc_sheet.jpg")


if __name__ == "__main__":
    main()
