"""Steering along the evidence direction under a conflicting hint.

The direction is fitted on the no-hint states, then applied during the
conflicting-hint forward pass. If the probe's evidence axis is still usable
under conflict, adding w to absent items should raise refusal and removing it
from present items should lower true-colour answers.
"""

import argparse
import json
import time

from PIL import Image

from .constants import DISTRACTOR
from .infer import DATA, ROOT, Scorer, build_user
from .steer import Steer, fit_direction, set_mode

KEYS = ("sufficient|wrong", "absent|a", "absent|b")


def prompts_for(row):
    acat, acolor = row["absent_category"], row["absent_hint"]
    return {
        "sufficient|wrong": build_user(row, "wrong"),
        "absent|a": build_user(row, "color", acat, acolor),
        "absent|b": build_user(row, "color", acat, DISTRACTOR[acolor]),
    }


def find_lm_head(model):
    for name, module in model.named_modules():
        if name.endswith("lm_head"):
            return module
    raise ValueError("no lm_head found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen"])
    ap.add_argument("--resize", type=int, default=1024)
    ap.add_argument("--run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "steer_conflict.jsonl"))
    ap.add_argument("--alphas", default="-0.5,-0.25,0.25,0.5")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[:args.limit]

    w, scale = fit_direction(args.run)
    print(f"direction fitted, median hidden norm {scale:.1f}", flush=True)

    if args.family == "qwen":
        scorer = Scorer(args.model)
        cand_names = scorer.cand_names

        def score(img, text):
            return scorer.score([img], text)[0]
    else:
        from .infer_gen import Gen
        scorer = Gen(args.model, resize=args.resize)
        cand_names = list(scorer.ids)

        def score(img, text):
            return scorer.score(img, text)

    st = Steer(scorer.model, w, scale, head=find_lm_head(scorer.model))
    settings = [("base", "none", 0.0), ("ablate", "ablate", 0.0)]
    settings += [(f"a{alpha:+.2f}", "add", alpha)
                 for alpha in [float(x) for x in args.alphas.split(",")]]

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(cand_names),
                                     "scale": scale,
                                     "alphas": args.alphas}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            prompts = prompts_for(row)
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for name, mode, alpha in settings:
                set_mode(st, mode, alpha * scale)
                for key in KEYS:
                    rec["probs"][f"{key}|{name}"] = score(img, prompts[key])
            set_mode(st, "none")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    st.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
