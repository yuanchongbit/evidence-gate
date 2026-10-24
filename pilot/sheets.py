"""Contact sheets for visually checking the automatic colour labels."""

import argparse
import json

from PIL import Image, ImageDraw, ImageFont

from .data import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "pool.jsonl"))
    ap.add_argument("--labels", default=str(DATA / "labels_pool.json"))
    ap.add_argument("--out-dir", default=str(DATA / "sheets"))
    ap.add_argument("--per-sheet", type=int, default=60)
    ap.add_argument("--cell", type=int, default=200)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    rows = [json.loads(l) for l in open(args.pool)]
    rows = [r for r in rows if labels.get(r["sample_id"], "skip") != "skip"]
    rows.sort(key=lambda r: (labels[r["sample_id"]], r["category"]))
    out = DATA / "sheets"
    out.mkdir(parents=True, exist_ok=True)

    cols, cell, bar = 6, args.cell, 26
    font = ImageFont.load_default(size=17)
    for s in range(0, len(rows), args.per_sheet):
        chunk = rows[s:s + args.per_sheet]
        nrow = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, nrow * (cell + bar)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            img = Image.open(DATA / "images" / r["file_name"]).convert("RGB")
            x, y, w, h = [int(v) for v in r["bbox"]]
            pad = 8
            crop = img.crop((max(0, x - pad), max(0, y - pad),
                             min(img.width, x + w + pad), min(img.height, y + h + pad)))
            crop.thumbnail((cell, cell))
            cx, cy = (i % cols) * cell, (i // cols) * (cell + bar)
            sheet.paste(crop, (cx + (cell - crop.width) // 2, cy + (cell - crop.height) // 2))
            lab = labels[r["sample_id"]]
            d.text((cx + 3, cy + cell + 3), f"{lab}  {r['category'][:14]}  {r['sample_id']}",
                   fill="black", font=font)
        path = out / f"sheet_{s // args.per_sheet:02d}.jpg"
        sheet.save(path, quality=88)
        print(path, len(chunk))


if __name__ == "__main__":
    main()
