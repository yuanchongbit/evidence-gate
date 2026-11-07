"""Resolution-matched 3B control.

The cross-model comparison also varies input resolution: Qwen-2.5-VL runs at
native pixels up to 1M, LLaVA-OneVision at a 512-px longest side. This reruns
the 3B battery at reduced pixel budgets so the same model spans the range and
the resolution contribution can be read off directly.
"""

import argparse
import json
import time

import numpy as np
from PIL import Image

from .constants import DISTRACTOR
from .infer import DATA, ROOT, Scorer, build_user

CONDS = ("sufficient|none", "sufficient|correct", "sufficient|wrong",
         "absent|none", "absent|a", "absent|b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    ap.add_argument("--tag", default="512")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[: args.limit]
    out = args.out or str(ROOT / "runs" / f"res3b_{args.tag}.jsonl")

    scorer = Scorer(args.model, max_pixels=args.max_pixels)
    hidden = {}
    t0 = time.time()
    with open(out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "kind": "resolution",
                                     "max_pixels": args.max_pixels,
                                     "candidates": ",".join(scorer.cand_names)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            acat, acolor = row["absent_category"], row["absent_hint"]
            prompts = {
                "sufficient|none": build_user(row, "none"),
                "sufficient|correct": build_user(row, "correct"),
                "sufficient|wrong": build_user(row, "wrong"),
                "absent|none": build_user(row, "none", acat),
                "absent|a": build_user(row, "color", acat, acolor),
                "absent|b": build_user(row, "color", acat, DISTRACTOR[acolor]),
            }
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": acat,
                   "absent_hint": acolor, "probs": {}}
            for key in CONDS:
                buf = []
                rec["probs"][key] = scorer.score([img], prompts[key],
                                                 batch_size=1, capture=buf)[0]
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    keys = list(hidden)
    np.savez_compressed(out.replace(".jsonl", ".npz"),
                        keys=np.array(keys), H=np.stack([hidden[k] for k in keys]))
    print("saved", out, "hidden", len(keys))


if __name__ == "__main__":
    main()
