"""Text-only control: same prompts, no image content.

Every prompt is paired with a uniform gray image of the same size, so the
hidden state can only carry the wording (category, hint) and not the evidence.
A probe that still separates present from absent on these states is reading a
lexical shortcut, not visual evidence availability.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import DISTRACTOR, HINTS
from .infer import DATA, ROOT, Scorer, build_user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "runs" / "blank.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    scorer = Scorer(args.model)
    hidden = {}
    t0 = time.time()
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(scorer.cand_names),
                                     "kind": "gray_image_control"}}) + "\n")
        for n, row in enumerate(rows, 1):
            ref = Image.open(ROOT / row["renders"]["orig"])
            gray = Image.new("RGB", ref.size, (128, 128, 128))
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "probs": {}}
            if row.get("absent_category"):
                rec["absent_category"] = row["absent_category"]
                rec["absent_hint"] = row["absent_hint"]
            for h in HINTS:
                key = f"sufficient|{h}"
                buf = []
                rec["probs"][key] = scorer.score([gray], build_user(row, h), capture=buf)[0]
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            if row.get("absent_category"):
                acat, acolor = row["absent_category"], row["absent_hint"]
                for tag, col in (("a", acolor), ("b", DISTRACTOR[acolor])):
                    key = f"absent|{tag}"
                    buf = []
                    rec["probs"][key] = scorer.score(
                        [gray], build_user(row, "color", acat, col), capture=buf)[0]
                    hidden[f"{row['sample_id']}|{key}"] = buf[0]
                key = "absent|none"
                buf = []
                rec["probs"][key] = scorer.score(
                    [gray], build_user(row, "none", acat), capture=buf)[0]
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    keys = list(hidden)
    H = np.stack([hidden[k] for k in keys])
    np.savez_compressed(out_path.with_suffix(".npz"), keys=np.array(keys), H=H)
    print("saved", out_path, H.shape)


if __name__ == "__main__":
    main()
