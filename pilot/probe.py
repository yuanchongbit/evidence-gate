"""Linear probe on the model's final hidden state for evidence availability.

The probe is trained on (present, absent) prompt pairs and evaluated with
GroupKFold on sample_id, so the same scene never appears in train and test.
The comparison of record is the model's own confidence from the same logits.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc

PAIRS = {
    "none": ("sufficient|none", "absent|none"),
    "hint": ("sufficient|wrong", "absent|a"),
    "correct": ("sufficient|correct", "absent|a"),
    "legacy": ("sufficient|none", "insufficient|none"),
}


def load(run_path):
    lines = [json.loads(l) for l in open(run_path)]
    meta = lines[0]["meta"]
    recs = [r for r in lines if "meta" not in r]
    z = np.load(run_path.replace(".jsonl", ".npz"), allow_pickle=False)
    H = dict(zip(z["keys"].tolist(), z["H"]))
    return recs, meta["candidates"].split(","), H


def dataset(recs, H, pos_key, neg_key, group_by="sample"):
    """group_by="sample" keeps scenes together; "prompt" keeps the prompted
    category together, so held-out folds contain categories never seen in
    training and a category-colour prior cannot carry the classifier."""
    X, y, g, conf, pk = [], [], [], [], []
    for r in recs:
        for key, lab in ((pos_key, 1), (neg_key, 0)):
            k = f"{r['sample_id']}|{key}"
            if k not in H:
                continue
            p = r["probs"][key]
            X.append(H[k].astype(np.float32))
            y.append(lab)
            if group_by == "prompt":
                g.append(r["absent_category"] if key == neg_key else r["category"])
            else:
                g.append(r["sample_id"])
            conf.append(float(np.max(p)))
            pk.append(float(1.0 - p[-1]))
    return np.stack(X), np.array(y), np.array(g), np.array(conf), np.array(pk)


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


def grouped_diff(a, b, y, g, n_boot=2000, seed=0):
    uniq = np.unique(g)
    idx_of = {k: np.where(g == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[k] for k in pick])
        vals.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cv_oof(X, y, g, C):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v5_261.jsonl")
    ap.add_argument("--pairs", default="none,hint")
    ap.add_argument("--Cs", default="0.0003,0.001,0.003,0.01,0.1")
    ap.add_argument("--fit-on", default="none", help="pair used to train the transfer probe")
    ap.add_argument("--group-by", default="sample", choices=["sample", "prompt"],
                    help="CV grouping: scene (sample) or prompted category (prompt)")
    args = ap.parse_args()
    recs, names, H = load(args.run)
    print(f"n={len(recs)} samples, hidden dim {H[next(iter(H))].shape}, "
          f"cv groups={args.group_by}")

    best = {}
    for name in args.pairs.split(","):
        pos, neg = PAIRS[name]
        X, y, g, conf, pk = dataset(recs, H, pos, neg, args.group_by)
        rows = []
        for C in [float(c) for c in args.Cs.split(",")]:
            oof = cv_oof(X, y, g, C)
            pt = auroc(oof, y)
            rows.append((pt, C, oof))
        rows.sort(reverse=True)
        pt, C, oof = rows[0]
        lo, hi = grouped_ci(oof, y, g)
        c = auroc(conf, y)
        dlo, dhi = grouped_diff(oof, conf, y, g)
        k = auroc(pk, y)
        klo, khi = grouped_diff(oof, pk, y, g)
        best[name] = (C, oof, X, y, g)
        print(f"\npair={name} ({pos} vs {neg})  n={len(y)}  groups={len(np.unique(g))}")
        print(f"  probe  AUROC {pt:.3f} [{lo:.3f}, {hi:.3f}]  (C={C:g})")
        print(f"  maxprob AUROC {c:.3f}   probe-maxprob {pt-c:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        print(f"  p_known AUROC {k:.3f}   probe-p_known {pt-k:+.3f} [{klo:+.3f}, {khi:+.3f}]")
        print("  by C:", ", ".join(f"{r[1]:g}:{r[0]:.3f}" for r in sorted(rows, key=lambda r: r[1])))

    if args.fit_on in best and args.pairs.count(",") >= 1:
        C, _, X, y, g = best[args.fit_on]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X, y)
        for name in args.pairs.split(","):
            if name == args.fit_on:
                continue
            Xh, yh, gh, confh, pkh = dataset(recs, H, *PAIRS[name], args.group_by)
            s = clf.decision_function(Xh)
            print(f"\ntransfer {args.fit_on}->{name}: probe {auroc(s, yh):.3f}  "
                  f"maxprob {auroc(confh, yh):.3f}  p_known {auroc(pkh, yh):.3f}")


if __name__ == "__main__":
    main()
