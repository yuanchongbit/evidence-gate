"""Full patching battery for the non-Qwen backbones.

Combines, in one pass per sample, the three controls that were previously
separate scripts for Qwen-family models:

  profile    paste the image's own no-hint absent state at layer L
  shuffled   paste another sample's no-hint absent state at layer L
  same-image paste the image's no-hint present (different question) state at L
  no-op      paste the conflict pass's own state back at L
  zero       paste a zero vector at L

Gemma loads through infer_gen.Gen (chat template); Phi through phi_infer.Gen
(native prompt). Output keys match patch3/patch5/patch6 so control_analyze.py
and analyze44.py can read them unchanged.
"""

import argparse
import json
import time

from PIL import Image

from .constants import CANDIDATES
from .infer import ROOT, build_user
from .patch3 import Tap


def find_layers(model):
    import torch

    cands = []
    for name, mod in model.named_modules():
        if (hasattr(mod, "layers") and isinstance(mod.layers, torch.nn.ModuleList)
                and len(mod.layers) >= 8):
            cands.append((name, mod))
    if not cands:
        raise RuntimeError("no decoder stack found")
    for name, mod in cands:
        if name.endswith("language_model"):
            return name, mod
    for name, mod in cands:
        if name == "model":
            return name, mod
    return min(cands, key=lambda x: len(x[0]))


def build_runner(args):
    if args.family == "phi":
        from .phi_infer import Gen
        g = Gen(args.model, resize=args.resize, num_crops=args.num_crops)

        def score(img, text):
            return g.score(img, text)

        return g, find_layers(g.model), score, ",".join(CANDIDATES)
    from .infer_gen import Gen
    g = Gen(args.model, resize=args.resize,
            trust_remote_code=args.trust_remote_code, dtype=args.dtype)

    def score(img, text):
        return g.score(img, text)

    return g, find_layers(g.model), score, ",".join(g.cand_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="gen", choices=["gen", "phi"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--num-crops", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--layers", default="8,16,24,30",
                    help="layers for the same/shuffled profile")
    ap.add_argument("--same-layers", default="",
                    help="layers for same-image present/shuffled controls (default: last two of --layers)")
    ap.add_argument("--noop-layers", default="",
                    help="layers for no-op/zero controls (default: last two of --layers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[:args.limit]
    warm = rows[args.skip - 1:args.skip] if args.skip else []
    rows = rows[args.skip:]

    profile = [int(x) for x in args.layers.split(",")]
    same = ([int(x) for x in args.same_layers.split(",")] if args.same_layers
            else profile[-2:])
    noop = ([int(x) for x in args.noop_layers.split(",")] if args.noop_layers
            else profile[-2:])
    all_layers = sorted(set(profile) | set(same) | set(noop))

    scorer, (stack_name, lm), score, cand = build_runner(args)
    taps = {L: Tap(lm.layers[L]) for L in all_layers}
    print("stack", stack_name, "n_layers", len(lm.layers),
          "profile", profile, "same", same, "noop", noop, flush=True)

    prev = None
    t0 = time.time()
    with open(args.out, "a" if args.append else "w") as f:
        if not args.append:
            f.write(json.dumps({"meta": {
                "model": args.model, "family": args.family, "resize": args.resize,
                "kind": "patch_full", "layers": profile, "same_layers": same,
                "noop_layers": noop, "candidates": cand}}) + "\n")
        for n, row in enumerate(warm + rows, 1):
            replay = n <= len(warm)
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            text_abs = build_user(row, "none", row["absent_category"])
            text_pres = build_user(row, "none")
            text_hint = build_user(row, "color", row["absent_category"],
                                   row["absent_hint"])
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}

            p_abs, cap_abs = capture(taps, all_layers, lambda: score(img, text_abs))
            p_pres, cap_pres = capture(taps, all_layers, lambda: score(img, text_pres))
            p_hint, cap_hint = capture(taps, noop, lambda: score(img, text_hint))
            rec["probs"]["absent|none"] = p_abs
            rec["probs"]["sufficient|none"] = p_pres
            rec["probs"]["absent|a"] = p_hint

            srcs = {"same_abs": cap_abs, "same_pres": cap_pres}
            if prev is not None:
                srcs["shuf_abs"], srcs["shuf_pres"] = prev
            for name in ("same_abs", "same_pres", "shuf_abs", "shuf_pres"):
                if name not in srcs:
                    continue
                use = profile if name in ("same_abs", "shuf_abs") else same
                for L in use:
                    taps[L].set(srcs[name][L])
                    rec["probs"][f"absent|a<-{name}@L{L}"] = score(img, text_hint)
                    taps[L].off()
            for L in noop:
                taps[L].set(cap_hint[L])
                rec["probs"][f"absent|a|noop@L{L}"] = score(img, text_hint)
                taps[L].off()
                taps[L].set([0.0] * len(cap_hint[L]))
                rec["probs"][f"absent|a|zero@L{L}"] = score(img, text_hint)
                taps[L].off()
            prev = (cap_abs, cap_pres)
            if replay:
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            k = n - len(warm)
            if k % 10 == 0:
                print(f"{k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    for tap in taps.values():
        tap.handle.remove()
    print("saved", args.out)


def capture(taps, ids, fn):
    for L in ids:
        taps[L].mode, taps[L].vec = "capture", None
    probs = fn()
    out = {}
    for L in ids:
        taps[L].off()
        out[L] = taps[L].vec
    return probs, out


if __name__ == "__main__":
    main()
