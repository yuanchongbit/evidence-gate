"""Strict-absence check on the main absent arm.

The main battery treats "absent" as annotation-level. This asks a plain yes/no
presence question about the same image and category, so the main results can be
recomputed on items the model itself judges absent.
"""

import argparse
import json

import numpy as np
from PIL import Image

from .infer import DATA, ROOT, Scorer
from .infer_gen import Gen


class YesNo(Scorer):
    def __init__(self, model_path):
        super().__init__(model_path)
        tok = self.processor.tokenizer
        self.cand_names = ["yes", "no"]
        self.cand_ids = {w: [tok.encode(" " + w, add_special_tokens=False)[0]]
                         for w in self.cand_names}


class YesNoGen(Gen):
    def __init__(self, path, resize=512):
        super().__init__(path, resize=resize)
        tok = self.proc.tokenizer
        self.yes = tok.encode(" yes", add_special_tokens=False)[0]
        self.no = tok.encode(" no", add_special_tokens=False)[0]

    def score(self, img, text, capture=None):
        import torch
        inputs = self.inputs([self.prep(img)], text)
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
        s = np.array([float(logits[self.yes]), float(logits[self.no])])
        s -= s.max()
        p = np.exp(s)
        return (p / p.sum()).tolist()


def question(cat):
    return f"Is there a {cat} in this image?\nAnswer yes or no."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "runs" / "presence_main.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[: args.limit]

    clf = YesNo(args.model) if args.family == "qwen" else YesNoGen(args.model, args.resize)
    with open(args.out, "w") as f:
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            if args.family == "qwen":
                p = clf.score([img], question(row["absent_category"]))[0][0]
            else:
                p = clf.score(img, question(row["absent_category"]))[0]
            f.write(json.dumps({"sample_id": row["sample_id"],
                                "absent_category": row["absent_category"],
                                "p_yes": p}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(n, "/", len(rows), flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
