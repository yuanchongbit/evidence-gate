"""Area-dominant color labelling from COCO instance masks.

The original hue-vote proposal let small saturated accents win over the
object's dominant paint (a white bus with an orange stripe became "orange").
Here the mask's pixels decide: achromatic share first, then the dominant
hue bin, with purity and margin gates that drop multi-coloured objects.
"""

import argparse
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from .constants import COLORS
from .data import DATA

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


def mask_from_seg(seg, w, h):
    m = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.array(m, bool)


def dominant_color(img, seg):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    mask = mask_from_seg(seg, img.width, img.height)
    if mask.sum() < 200:
        return None, {"reason": "small_mask"}
    px = arr[mask]
    h, s, v = rgb_to_hsv_array(px)
    achr = s < ACHROMATIC_S
    a_frac = float(achr.mean())

    if a_frac >= 0.6:
        vmed = float(np.median(v[achr]))
        if vmed >= 0.72:
            color = "white"
        elif vmed >= 0.42:
            color = "gray"
        elif vmed <= 0.25:
            color = "black"
        else:
            return None, {"reason": "achromatic_ambiguous", "a_frac": a_frac, "vmed": vmed}
        margin = min(abs(vmed - 0.72), abs(vmed - 0.42), abs(vmed - 0.25))
        return color, {"a_frac": a_frac, "vmed": vmed, "purity": a_frac, "margin": margin}

    chrom = ~achr
    c_frac = float(chrom.mean())
    if c_frac < 0.4:
        return None, {"reason": "mixed", "a_frac": a_frac}

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
        return None, {"reason": "hue_impure", "a_frac": a_frac,
                      "c_frac": c_frac, "purity": purity,
                      "margin": margin, "top": color}

    # dark warm pixels are brown, not the bright hue bin
    vmed = float(np.median(v[chrom]))
    if color in ("orange", "yellow", "red"):
        warm = chrom & ((h >= 10) & (h < 50))
        if warm.sum() >= 50:
            warm_v = float(np.median(v[warm]))
            if warm_v < 0.45:
                color = "brown"
                vmed = warm_v
    if color not in COLORS:
        return None, {"reason": "bin_missing", "top": color}
    return color, {"a_frac": a_frac, "c_frac": c_frac,
                   "purity": purity, "margin": margin, "vmed": vmed}


def label_candidates(candidates, labels_path, min_purity=0.6, keep=6):
    KEEP = {"red", "blue", "green", "yellow", "black", "white"}
    labels = {}
    drops = Counter()
    for r in candidates:
        img = Image.open(DATA / "images" / r["file_name"])
        color, info = dominant_color(img, r["segmentation"])
        if color is None or color not in KEEP:
            labels[r["sample_id"]] = "skip"
            drops[info.get("reason", f"out_of_keep:{color}")] += 1
        elif info.get("purity", 1.0) < min_purity:
            labels[r["sample_id"]] = "skip"
            drops["low_purity"] += 1
        else:
            labels[r["sample_id"]] = color
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=0)
    kept = Counter(v for v in labels.values() if v != "skip")
    print("labels ->", labels_path)
    print("kept", sum(kept.values()), "of", len(candidates), dict(kept))
    print("drop reasons", drops.most_common())
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(DATA / "candidates.jsonl"))
    ap.add_argument("--out", default=str(DATA / "labels_auto.json"))
    ap.add_argument("--min-purity", type=float, default=0.6)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.candidates)]
    label_candidates(rows, args.out, args.min_purity)


if __name__ == "__main__":
    main()
