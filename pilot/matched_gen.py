"""Category-matched run for the model-agnostic Gen wrapper (LLaVA, Gemma, ...).

Same manifest and protocol as matched.py, but uses infer_gen.Gen so the
category-matched control can be replicated outside the Qwen2.5-VL loader.
"""

import argparse
import json

import numpy as np
from PIL import Image

from .infer import ROOT, build_user
from .infer_gen import Gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    g = Gen(args.model, resize=args.resize,
            trust_remote_code=args.trust_remote_code, dtype=args.dtype)
    hidden = {}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(g.cand_names),
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
