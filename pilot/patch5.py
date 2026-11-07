"""Same-image opposite-question patching control.

patch3/patch4 paste a no-hint absent state into the absent conflict pass. The
shuffled source is label-matched (another absent scene), so it separates "this
sample's evidence" from "an absent-scene prior" but not from any no-hint state.

Here every source comes from the same image, and we add a source whose question
is the present category instead of the absent category:

  same_abs   the image's absent/no-hint state       (same question, refusing)
  same_pres  the image's present/no-hint state      (different question, answering)
  shuf_abs   another sample's absent/no-hint state
  shuf_pres  another sample's present/no-hint state

same_abs vs same_pres holds the image fixed and varies the question and its
answer; shuf_* adds a scene change on top.
"""

import argparse
import json
import time

from PIL import Image

from .infer import DATA, ROOT, Scorer, build_user
from .patch3 import Tap

SOURCES = ("same_abs", "shuf_abs", "same_pres", "shuf_pres")


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
    ap.add_argument("--out", default=str(ROOT / "runs" / "patch5.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[:args.limit]
    # shuffled sources come from the previous sample; on resume, replay it
    warm = rows[args.skip - 1:args.skip] if args.skip else []
    rows = rows[args.skip:]

    scorer, layers, score = build_runner(args)
    layer_ids = [int(x) for x in args.layers.split(",")]
    taps = {L: Tap(layers[L]) for L in layer_ids}
    cand_names = getattr(scorer, "cand_names", None) or list(scorer.ids)
    print("n_layers", len(layers), "probe", layer_ids, flush=True)

    prev = None
    t0 = time.time()
    with open(args.out, "a" if args.append else "w") as f:
        if not args.append:
            f.write(json.dumps({"meta": {
                "model": args.model, "family": args.family, "resize": args.resize,
                "kind": "patching5", "layers": layer_ids,
                "candidates": ",".join(cand_names)}}) + "\n")
        for n, row in enumerate(warm + rows, 1):
            replay = n <= len(warm)
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            text_abs_none = build_user(row, "none", row["absent_category"])
            text_pres_none = build_user(row, "none")
            text_hint = build_user(row, "color", row["absent_category"],
                                   row["absent_hint"])
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}

            cap_abs, cap_pres = {}, {}
            for L, tap in taps.items():
                tap.mode, tap.vec = "capture", None
            rec["probs"]["absent|none"] = score(img, text_abs_none)
            for L, tap in taps.items():
                tap.off()
                cap_abs[L], tap.vec = tap.vec, None
            for L, tap in taps.items():
                tap.mode = "capture"
            rec["probs"]["sufficient|none"] = score(img, text_pres_none)
            for L, tap in taps.items():
                tap.off()
                cap_pres[L] = tap.vec

            rec["probs"]["absent|a"] = score(img, text_hint)
            srcs = {"same_abs": cap_abs, "same_pres": cap_pres}
            if prev is not None:
                srcs["shuf_abs"], srcs["shuf_pres"] = prev
            for name in SOURCES:
                if name not in srcs:
                    continue
                for L, tap in taps.items():
                    tap.set(srcs[name][L])
                    rec["probs"][f"absent|a<-{name}@L{L}"] = score(img, text_hint)
                    tap.off()
            prev = (cap_abs, cap_pres)
            if replay:
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n = n - len(warm)
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
