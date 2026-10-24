"""Second task: spatial side instead of colour.

Same 2x3 design with an answer that has nothing to do with colour. The gold
side is computed from the COCO bbox, the conflicting hint asserts the other
side, and the absent arm asks about a category the annotations place outside
the image. If the gate/code dissociation is about asserted answers rather than
about colour words, it replicates here.
"""

import argparse
import json
import random
import time

import numpy as np
from PIL import Image

from .infer import ROOT, Scorer


class SideScorer(Scorer):
    def __init__(self, model_path, dtype="bfloat16"):
        super().__init__(model_path, dtype=dtype)
        tok = self.processor.tokenizer
        self.cand_names = ["left", "right", "unknown"]
        ids = {}
        for word in self.cand_names:
            forms = [tok.encode(" " + word, add_special_tokens=False),
                     tok.encode(word, add_special_tokens=False)]
            single = [f[0] for f in forms if len(f) == 1]
            if not single:
                raise ValueError(f"{word!r} is not a single token")
            ids[word] = sorted(set(single))
        self.cand_ids = ids


def question_cat(cat, side=None):
    prefix = f"Hint: the {cat} is on the {side}.\n" if side else ""
    return f"{prefix}Which half of the image is the {cat} in, left or right?\nAnswer with one word."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "runs" / "task2.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[:args.limit]
    rng = random.Random(args.seed)
    scorer = SideScorer(args.model)
    t0 = time.time()
    hidden = {}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(scorer.cand_names),
                                     "kind": "side"}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            x, y, w, h = row["bbox"]
            gold = "left" if x + w / 2 < img.width / 2 else "right"
            other = "right" if gold == "left" else "left"
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": gold, "probs": {}}
            for key, text in (("sufficient|none", question_cat(row["category"])),
                              ("sufficient|wrong",
                               question_cat(row["category"], other))):
                buf = []
                rec["probs"][key] = scorer.score([img], text, capture=buf)[0]
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            acat = row["absent_category"]
            aside = rng.choice(["left", "right"])
            rec["absent_category"] = acat
            rec["absent_side"] = aside
            for key, text in (("absent|none", question_cat(acat)),
                              ("absent|a", question_cat(acat, aside))):
                buf = []
                rec["probs"][key] = scorer.score([img], text, capture=buf)[0]
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"), keys=np.array(keys),
                        H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
