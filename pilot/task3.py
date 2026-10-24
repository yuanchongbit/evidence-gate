"""Third task: count instead of colour.

All v5 images contain exactly one annotated instance, so the gold answer is
"one" and the conflicting hint asserts two. The absent arm asks the same
question about a category the annotations place outside the image.
"""

import argparse
import json
import time

import numpy as np
from PIL import Image

from .infer import ROOT, Scorer


class CountScorer(Scorer):
    def __init__(self, model_path, dtype="bfloat16"):
        super().__init__(model_path, dtype=dtype)
        tok = self.processor.tokenizer
        self.cand_names = ["one", "two", "three", "unknown"]
        ids = {}
        for word in self.cand_names:
            forms = [tok.encode(" " + word, add_special_tokens=False),
                     tok.encode(word, add_special_tokens=False)]
            single = [f[0] for f in forms if len(f) == 1]
            if not single:
                raise ValueError(f"{word!r} is not a single token")
            ids[word] = sorted(set(single))
        self.cand_ids = ids


def question(cat, hint=None):
    prefix = f"Hint: there are {hint} {cat} in this image.\n" if hint else ""
    return f"{prefix}How many {cat} are in this image?\nAnswer with one word."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "runs" / "task3.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[:args.limit]
    scorer = CountScorer(args.model)
    t0 = time.time()
    hidden = {}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(scorer.cand_names),
                                     "kind": "count"}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            cat, acat = row["category"], row["absent_category"]
            rec = {"sample_id": row["sample_id"], "category": cat,
                   "gold": "one", "absent_category": acat, "probs": {}}
            for key, text in (("sufficient|none", question(cat)),
                              ("sufficient|wrong", question(cat, "two")),
                              ("absent|none", question(acat)),
                              ("absent|a", question(acat, "two"))):
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
