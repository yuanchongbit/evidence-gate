"""No-op patching control.

patch5 pastes states across passes, which changes both the vector and the pass.
Here the pasted vector is the one the same pass produced at that layer, so a
correctly wired hook must return the base distribution exactly. Comparison
against a zero-paste (also through the same hook) shows the hook is active and
that the no-op is not vacuous.
"""

import argparse
import json
import time

from PIL import Image

from .infer import DATA, ROOT, Scorer, build_user
from .patch3 import Tap


def build_runner(args):
    if args.family == "qwen":
        scorer = Scorer(args.model)
        lm = scorer.model.model.language_model

        def score(img, text):
            return scorer.score([img], text)[0]
    else:
        from .infer_gen import Gen
        from .patch4 import find_layers
        g = Gen(args.model, resize=args.resize)
        _, lm = find_layers(g.model)
        scorer = g

        def score(img, text):
            return g.score(img, text)
    return scorer, lm.layers, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--layers", default="24,32")
    ap.add_argument("--out", default=str(ROOT / "runs" / "patch6_noop.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[: args.limit]

    scorer, layers, score = build_runner(args)
    layer_ids = [int(x) for x in args.layers.split(",")]
    taps = {L: Tap(layers[L]) for L in layer_ids}
    cand = getattr(scorer, "cand_names", None) or list(scorer.ids)
    print("n_layers", len(layers), "probe", layer_ids, flush=True)

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "family": args.family,
                                     "resize": args.resize, "kind": "patch6",
                                     "layers": layer_ids,
                                     "candidates": ",".join(cand)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            text = build_user(row, "color", row["absent_category"], row["absent_hint"])
            rec = {"sample_id": row["sample_id"],
                   "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            cap = {}
            for L, tap in taps.items():
                tap.mode, tap.vec = "capture", None
            rec["probs"]["absent|a"] = score(img, text)
            for L, tap in taps.items():
                tap.off()
                cap[L] = tap.vec
            for L, tap in taps.items():
                tap.set(cap[L])
                rec["probs"][f"absent|a|noop@L{L}"] = score(img, text)
                tap.off()
                tap.set([0.0] * len(cap[L]))
                rec["probs"][f"absent|a|zero@L{L}"] = score(img, text)
                tap.off()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
