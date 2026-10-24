"""Analyse the VQA-v2 colour replication: does the frozen 261-set probe keep
tracking image evidence when a misleading hint is added, while output signals
collapse? Risk-coverage on plain (no-hint) accuracy as a secondary check."""
import argparse
import json

import numpy as np

from .constants import DISTRACTOR
from .metrics import auroc
from .probe import grouped_ci, grouped_diff

HINTS = ("none", "correct", "wrong")
COVS = (1.0, 0.9, 0.7, 0.5)


def load(run):
    return [json.loads(l) for l in open(run)]


def col(recs, hint, key):
    return np.array([r["res"][hint][key] for r in recs], dtype=float)


def correct(recs, hint):
    return np.array([r["res"][hint]["pred"] == r["gold"] for r in recs], dtype=float)


def rc(score, ok, covs=COVS):
    order = np.argsort(-np.asarray(score, dtype=float))
    out = {}
    for cov in covs:
        k = max(1, int(round(cov * len(order))))
        out[f"{cov:.1f}"] = float(1.0 - ok[order[:k]].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ext_vqa_color.jsonl")
    args = ap.parse_args()
    recs = load(args.run)
    g = np.array([str(r["image_id"]) for r in recs])
    print(f"{len(recs)} colour questions, {len(np.unique(g))} images")

    accs = {}
    for h in HINTS:
        ok = correct(recs, h)
        accs[h] = ok
        pu = col(recs, h, "p_unk")
        pr = col(recs, h, "probe")
        pred = np.array([r["res"][h]["pred"] for r in recs])
        wrong = np.array([DISTRACTOR[r["gold"]] for r in recs])
        print(f"  {h:8s} acc {ok.mean():.3f}  p_unk {pu.mean():.3f}  "
              f"maxp {col(recs, h, 'maxprob').mean():.3f}  probe {pr.mean():+.3f}  "
              f"probe>0 {np.mean(pr > 0):.3f}  pred=unknown {np.mean(pred == 'unknown'):.3f}  "
              f"pred=distractor {np.mean(pred == wrong):.3f}")

    print("\nprobe stability (image evidence) vs output-signal stability")
    for key in ("probe", "p_unk", "maxprob"):
        a, b = col(recs, "none", key), col(recs, "wrong", key)
        print(f"  {key:8s} r(none,wrong) {np.corrcoef(a, b)[0, 1]:.3f}  "
              f"mean shift {b.mean() - a.mean():+.3f}")
    pn = col(recs, "none", "probe")
    print(f"  probe(none)>0 among none-correct: {np.mean(pn[accs['none'] > 0] > 0):.3f}")

    print("\nAUROC for being correct under the wrong hint")
    target = accs["wrong"]
    for src in ("none", "wrong"):
        for key, sign in (("probe", 1.0), ("maxprob", 1.0), ("p_unk", -1.0)):
            s = sign * col(recs, src, key)
            pt = auroc(s, target)
            lo, hi = grouped_ci(s, target, g)
            print(f"  {src:6s} {key:8s} {pt:.3f} [{lo:.3f},{hi:.3f}]")
    for src in ("none", "wrong"):
        pr = col(recs, src, "probe")
        pk = -col(recs, src, "p_unk")
        dlo, dhi = grouped_diff(pr, pk, target, g)
        print(f"  {src:6s} probe - p_known {auroc(pr, target) - auroc(pk, target):+.3f} "
              f"[{dlo:+.3f},{dhi:+.3f}]")

    print("\nrisk (1-acc) at coverage, no-hint answers")
    sig = {"probe": col(recs, "none", "probe"),
           "maxprob": col(recs, "none", "maxprob"),
           "p_known": -col(recs, "none", "p_unk")}
    risk = {}
    for k, s in sig.items():
        risk[k] = rc(s, accs["none"])
        print(f"  {k:8s} " + "  ".join(f"{risk[k][f'{c:.1f}']:.3f}" for c in COVS))

    summary = {
        "n": len(recs),
        "acc": {h: float(accs[h].mean()) for h in HINTS},
        "probe_stability_r": float(np.corrcoef(pn, col(recs, "wrong", "probe"))[0, 1]),
        "auroc_correct_under_wrong": {
            f"{src}|{key}": float(auroc(col(recs, src, key), target))
            for src in ("none", "wrong") for key in ("probe", "maxprob", "p_unk")},
        "risk": risk,
    }
    with open(args.run.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved", args.run.replace(".jsonl", "_summary.json"))


if __name__ == "__main__":
    main()
