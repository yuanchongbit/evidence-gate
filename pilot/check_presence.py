"""Manipulation check: is the object still visible after removal?

Asks a plain yes/no existence question about the same image the colour question
uses. A colour question presupposes the object, so it cannot tell us whether
removal worked; this can.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from .infer import DATA, ROOT, Scorer


class YesNo(Scorer):
    def __init__(self, model_path, dtype="bfloat16"):
        super().__init__(model_path, dtype=dtype)
        tok = self.processor.tokenizer
        self.cand_names = ["yes", "no"]
        self.cand_ids = {w: [tok.encode(" " + w, add_special_tokens=False)[0]]
                         for w in self.cand_names}


def question(row):
    return f"Is there a {row['category']} in this image?\nAnswer yes or no."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "runs" / "presence_v5.jsonl"))
    ap.add_argument("--style-b", default="", help="dir with variant renders")
    ap.add_argument("--style-suffix", default="insufficient_b",
                    help="filename suffix before .jpg in --style-b")
    ap.add_argument("--style-key", default="p_yes_styleb")
    ap.add_argument("--control", default="",
                    help="json mapping sample_id -> category verified absent from the image")
    ap.add_argument("--only-control", action="store_true")
    ap.add_argument("--only-styleb", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    control = json.load(open(args.control)) if args.control else {}
    done = {}
    if args.resume and Path(args.out).exists():
        done = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.out)}
    clf = YesNo(args.model)
    with open(args.out, "w") as f:
        for n, row in enumerate(rows, 1):
            rec = dict(done.get(row["sample_id"], {}))
            rec.setdefault("sample_id", row["sample_id"])
            rec.setdefault("category", row["category"])
            if not args.only_control and not args.only_styleb:
                for tag in ("orig", "insufficient"):
                    img = Image.open(ROOT / row["renders"][tag]).convert("RGB")
                    rec[f"p_yes_{tag}"] = clf.score([img], question(row))[0][0]
            if args.style_b and not args.only_control:
                p = Path(args.style_b) / f'{row["sample_id"]}_{args.style_suffix}.jpg'
                img = Image.open(p).convert("RGB")
                rec[args.style_key] = clf.score([img], question(row))[0][0]
            if row["sample_id"] in control and "p_yes_control" not in rec:
                c = control[row["sample_id"]]
                img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
                rec["p_yes_control"] = clf.score(
                    [img], f"Is there a {c} in this image?\nAnswer yes or no.")[0][0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(n, "/", len(rows), flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
