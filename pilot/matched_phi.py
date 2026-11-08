"""Category-matched run for the native-prompt Phi-3.5-vision loader."""

import argparse
import json

import numpy as np
from PIL import Image

from .constants import CANDIDATES
from .infer import ROOT, build_user
from .phi_infer import Gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--num-crops", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    g = Gen(args.model, resize=args.resize, num_crops=args.num_crops)
    hidden = {}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(CANDIDATES),
                                     "kind": "category_matched", "gen": True}}) + "\n")
        for n, r in enumerate(rows, 1):
            img = Image.open(ROOT / r["renders"]["orig"]).convert("RGB")
            buf = []
            p = g.score(img, build_user(r, "none"), buf)
            hidden[r["sample_id"]] = buf[0]
            rec = {k: r[k] for k in ("pair_id", "side", "sample_id", "category", "present")}
            rec["probs"] = {"none": p}

            text = (f"Hint: the {r['category']} is {r['hint_color']}.\n"
                    f"What color is the {r['category']}?\nAnswer with one word.")
            buf = []
            rec["probs"]["conflict"] = g.score(img, text, buf)
            hidden[f"{r['sample_id']}|conflict"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(n, "/", len(rows), flush=True)

    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"), keys=np.array(keys),
                        H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
