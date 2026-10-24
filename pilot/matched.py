"""Category-matched control for the evidence probe.

The main manifest asks about a different category in the present and absent
arms, so a probe could separate the arms from the question text alone (the
gray-image control does exactly that). Here the prompt and the prompted
category are identical on both sides and only the image changes: one side is
the v5 image that contains the object, the other is another v5 image where the
same category is annotated absent. Both images come from the same pool (one
large centred COCO instance), so scene statistics are matched too.

Cross-validation groups by prompted category, so held-out folds contain
categories the probe never saw.

Sub-modes:
  --build    write data/manifest_matched.jsonl (local, needs COCO annotations)
  --presence verify the negative images really lack the category (GPU)
  (default)  run the colour question on both sides and analyse
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np

from .constants import DISTRACTOR
from .metrics import auroc


def build(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    coco = json.load(open(args.annot))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_cats = defaultdict(set)
    for a in coco["annotations"]:
        img_cats[a["image_id"]].add(cat_name[a["category_id"]])

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    by_cat = {c: rs for c, rs in by_cat.items() if len(rs) >= args.min_pos}

    rng = random.Random(args.seed)
    out = []
    for cat, rs in sorted(by_cat.items()):
        donors = [d for d in rows if cat not in img_cats[d["image_id"]]]
        if not donors:
            continue
        for r in rs:
            d = rng.choice(donors)
            hint = DISTRACTOR[r["gold"]]
            out.append({"pair_id": r["sample_id"], "side": "pos",
                        "sample_id": f"pos_{r['sample_id']}",
                        "category": cat, "present": 1, "hint_color": hint,
                        "image_id": r["image_id"],
                        "renders": {"orig": r["renders"]["orig"]}})
            out.append({"pair_id": r["sample_id"], "side": "neg",
                        "sample_id": f"neg_{r['sample_id']}",
                        "category": cat, "present": 0, "hint_color": hint,
                        "image_id": d["image_id"],
                        "renders": {"orig": d["renders"]["orig"]}})
    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    cats = sorted(by_cat)
    print(f"{len(out)} rows, {len(out)//2} pairs, {len(cats)} categories "
          f"(min {min(len(by_cat[c]) for c in cats)} pos/cat) -> {args.out}")


def presence(args):
    from PIL import Image
    from .check_presence import YesNo
    from .infer import DATA, ROOT
    rows = [json.loads(l) for l in open(args.manifest)]
    neg = [r for r in rows if not r["present"]]
    clf = YesNo(args.model)
    out = []
    with open(args.presence_out, "w") as f:
        for n, r in enumerate(neg, 1):
            img = Image.open(ROOT / r["renders"]["orig"]).convert("RGB")
            q = f"Is there a {r['category']} in this image?\nAnswer yes or no."
            p = clf.score([img], q)[0][0]
            f.write(json.dumps({"sample_id": r["sample_id"],
                                "category": r["category"], "p_yes": p}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(n, "/", len(neg), flush=True)
    print("saved", args.presence_out)


def run(args):
    import time
    from PIL import Image
    from .infer import ROOT, Scorer, build_user
    rows = [json.loads(l) for l in open(args.manifest)]
    scorer = Scorer(args.model)
    t0 = time.time()
    hidden = {}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(scorer.cand_names),
                                     "kind": "category_matched"}}) + "\n")
        for n, r in enumerate(rows, 1):
            img = Image.open(ROOT / r["renders"]["orig"]).convert("RGB")
            buf = []
            p = scorer.score([img], build_user(r, "none"), capture=buf)[0]
            hidden[r["sample_id"]] = buf[0]
            rec = {k: r[k] for k in ("pair_id", "side", "sample_id", "category", "present")}
            rec["probs"] = {"none": p}
            if args.conflict:
                text = (f"Hint: the {r['category']} is {r['hint_color']}.\n"
                        f"What color is the {r['category']}?\nAnswer with one word.")
                buf = []
                rec["probs"]["conflict"] = scorer.score([img], text, capture=buf)[0]
                hidden[f"{r['sample_id']}|conflict"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"), keys=np.array(keys),
                        H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out)


def grouped_ci(scores, y, g, n_boot=2000, seed=0):
    uniq = np.unique(g)
    idx_of = {k: np.where(g == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[k] for k in pick])
        vals.append(auroc(scores[idx], y[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cv_oof(X, y, g, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def hidden_of(H, sid, key):
    k = f"{sid}|{key}"
    return H[k] if k in H else H[sid]


def analyze(args):
    lines = [json.loads(l) for l in open(args.run)]
    recs = [r for r in lines if "meta" not in r]
    z = np.load(args.run.replace(".jsonl", ".npz"))
    H = dict(zip(z["keys"].tolist(), z["H"]))
    y = np.array([r["present"] for r in recs])
    cat = np.array([r["category"] for r in recs])
    pair = np.array([r["pair_id"] for r in recs])
    print(f"n={len(recs)} pairs={len(np.unique(pair))} categories={len(np.unique(cat))}")

    for key in [k for k in ("none", "conflict") if k in recs[0]["probs"]]:
        X = np.stack([hidden_of(H, r["sample_id"], key) for r in recs]).astype(np.float32)
        pk = np.array([1.0 - r["probs"][key][-1] for r in recs])
        mp = np.array([float(np.max(r["probs"][key])) for r in recs])
        print(f"\n=== condition {key}")
        for label, g in (("by pair", pair), ("by category", cat)):
            report(X, y, g, pk, mp, label)


def report(X, y, g, pk, mp, label):
    best = None
    for C in [0.0003, 0.001, 0.003, 0.01, 0.1]:
        oof = cv_oof(X, y, g, C)
        pt = auroc(oof, y)
        if best is None or pt > best[0]:
            best = (pt, C, oof)
    pt, C, oof = best
    lo, hi = grouped_ci(oof, y, g)
    print(f"CV {label} ({len(np.unique(g))} groups, C={C:g})")
    print(f"  probe  AUROC {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  p_known AUROC {auroc(pk, y):.3f}   maxprob {auroc(mp, y):.3f}")
    print(f"  probe-p_known {pt - auroc(pk, y):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--annot", default=str(DATA / "raw" / "annotations" / "instances_val2017.json"))
    ap.add_argument("--out", default=str(DATA / "manifest_matched.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--run", default=str(ROOT / "runs" / "matched.jsonl"))
    ap.add_argument("--min-pos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--presence", action="store_true")
    ap.add_argument("--presence-out",
                    default=str(ROOT / "runs" / "presence_matched.jsonl"))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--conflict", action="store_true",
                    help="also run/analyse a hint that asserts a wrong colour")
    args = ap.parse_args()
    if args.build:
        build(args)
    elif args.presence:
        presence(args)
    elif args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
