"""Six-form battery, extra cell: an agreeing (correct) hint with evidence present.

The main six-form battery has sufficient|none and sufficient|wrong but no
sufficient|correct, so it cannot separate "a hint sentence was added" from
"the hint conflicts with the image".  This runs the missing cell for every
form, keyed form|sufficient|correct, so v_c = mean(sufficient correct) -
mean(absent a) can be compared with v0 and v_a on the same scenes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CANDIDATES
from .infer import DATA, ROOT
from .prompt_forms import FORMS, FormScorer, set_form, user_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "pf_correct.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")][:: args.stride]
    if args.limit:
        rows = rows[: args.limit]

    model = FormScorer(args.model)
    hidden = {}
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "forms": list(FORMS),
                                     "stride": args.stride, "cell": "correct",
                                     "candidates": ",".join(CANDIDATES)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for form in FORMS:
                set_form(model, form, row, "correct", row["category"], row["gold"])
                text = user_text(form, row, "correct", row["category"], row["gold"])
                buf = []
                key = f"{form}|sufficient|correct"
                rec["probs"][key] = model.score([img], text, capture=buf)[0]
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
