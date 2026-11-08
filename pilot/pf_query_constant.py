"""Constant-colour queried-category control.

The existing query-specificity control compares three axes: a neutral extra
sentence, an assertion about another object, and the nl conflict axis.  The
first two hold one sentence fixed across evidence arms; the conflict axis also
changes the asserted colour between arms.  This run adds the missing cell: a
sentence that names the queried category in each arm and states the *same*
colour in both arms (the scene's absent-category colour).  The only difference
from the other-object control is therefore which object the sentence is about.

  v0  no-hint evidence axis (main battery)
  vn  neutral extra sentence (no object, no colour)
  vo  assertion about another object, constant colour
  vq  assertion about the queried category, constant colour   <- this run
  va  nl conflict axis (main battery; colour changes between arms)
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

QUERY = "A colleague mentioned that the {category} is {colour}."


def prompts(row):
    out = []
    for ev, cat in (("absent", row["absent_category"]),
                    ("sufficient", row["category"])):
        sent = QUERY.format(category=cat, colour=row["absent_hint"])
        out.append(("query", ev, "none", sent, cat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=("qwen", "gen"), default="qwen")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "pf_query_constant.jsonl"))
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
                                     "query": QUERY,
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
