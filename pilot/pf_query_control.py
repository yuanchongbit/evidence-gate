"""Query-specificity control for the evidence-axis rotation.

Two colleague-report sentences are added to both evidence arms:

  neutral: "A colleague mentioned that the room is quiet."
           (an extra sentence, but no object, no colour, not the query)
  other:   "A colleague mentioned that the remote is {absent_hint}."
           (an assertion naming another object and the same colour the
           nl form asserts about the queried category)

The evidence axis of each control is mean(sufficient) - mean(absent) within
the same sentence.  With the no-hint axis from the main battery and the
queried-category axis from the nl form, this separates three sources of the
rotation: the extra sentence (neutral), content about another object (other),
and content about the queried category (nl).  "remote" is one of the few
COCO categories that appears neither as a queried category nor as an absent
category in the manifest, so it cannot coincide with the query.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CANDIDATES
from .infer import DATA, ROOT, SYSTEM
from .prompt_forms import FormScorer, question

CONTROLS = {
    "neutral": "A colleague mentioned that the room is quiet.",
    "other": "A colleague mentioned that the remote is {colour}.",
}


def prompts(row):
    out = []
    for form, sent in CONTROLS.items():
        if form == "other":
            sent = sent.format(colour=row["absent_hint"])
        out.append((form, "absent", "none", sent, row["absent_category"]))
        out.append((form, "sufficient", "none", sent, row["category"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=("qwen", "gen"), default="qwen")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "pf_query_control.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")][:: args.stride]
    if args.limit:
        rows = rows[: args.limit]

    if args.backend == "qwen":
        model = FormScorer(args.model, dtype=args.dtype)
    else:
        from .prompt_forms_gen import FormGen
        model = FormGen(args.model, resize=args.resize, dtype=args.dtype)

    hidden = {}
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "backend": args.backend,
                                     "controls": CONTROLS,
                                     "dtype": args.dtype,
                                     "stride": args.stride,
                                     "candidates": ",".join(CANDIDATES)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for form, ev, hint, sent, cat in prompts(row):
                model.form = "nl"
                model.system = SYSTEM
                model.turn = ""
                text = f"{sent} {question(cat)}"
                buf = []
                key = f"{form}|{ev}|{hint}"
                if args.backend == "qwen":
                    rec["probs"][key] = model.score([img], text, capture=buf)[0]
                else:
                    rec["probs"][key] = model.score(img, text, capture=buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample "
                      f"eta {(len(rows) - n) * rate / 60:.1f}min", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"),
                        keys=np.array(keys), H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out, "hidden", len(keys))


if __name__ == "__main__":
    main()
