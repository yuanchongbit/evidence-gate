"""Risk-coverage of answer/abstain gates under no / correct / conflicting hints."""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .probe import PAIRS, load


def oof_scores(X, y, g, C=0.1):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def gather(recs, H, pos, neg):
    X, y, g, conf, pk, margin = [], [], [], [], [], []
    for r in recs:
        for key, lab in ((pos, 1), (neg, 0)):
            k = f"{r['sample_id']}|{key}"
            if k not in H:
                continue
            p = np.asarray(r["probs"][key], float)
            X.append(H[k].astype(np.float32))
            y.append(lab)
            g.append(r["sample_id"])
            conf.append(float(p.max()))
            pk.append(float(1.0 - p[-1]))
            margin.append(float(p[:-1].max() - p[-1]))
    return (np.stack(X), np.array(y), np.array(g), np.array(conf), np.array(pk),
            np.array(margin))


def at_coverage(score, y, cov):
    k = int(round(cov * len(score)))
    order = np.argsort(-score, kind="stable")
    ans = np.zeros(len(score), bool)
    ans[order[:k]] = True
    absent = y == 0
    suf = y == 1
    uar = float(ans[absent].mean())
    oar = float(1.0 - ans[suf].mean())
    prec = float(suf[ans].mean()) if ans.any() else float("nan")
    return uar, oar, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v5_261.jsonl")
    ap.add_argument("--pairs", default="none,hint,correct")
    ap.add_argument("--cov", default="0.5,0.7,0.9")
    args = ap.parse_args()
    recs, names, H = load(args.run)
    covs = [float(c) for c in args.cov.split(",")]
    print(f"n={len(recs)}")
    for name in args.pairs.split(","):
        X, y, g, conf, pk, margin = gather(recs, H, *PAIRS[name])
        pr = oof_scores(X, y, g)
        print(f"\n== {name}: n={len(y)} (absent {int((y == 0).sum())})")
        print("  cov   signal     UAR(fab)  OAR(refuse-suf)  answered-precision")
        for c in covs:
            for lab, s in (("probe", pr), ("p_known", pk), ("maxprob", conf),
                           ("margin", margin)):
                u, o, p = at_coverage(s, y, c)
                print(f"  {c:.1f}   {lab:<8s}   {u:.3f}     {o:.3f}            {p:.3f}")


if __name__ == "__main__":
    main()
