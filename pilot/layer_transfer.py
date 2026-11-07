"""Per-layer evidence readout with a single estimator.

The appendix previously refitted a probe at every layer, which confounds depth
with a new fit. Here one logistic-regression estimator is fitted on the final
layer (per CV fold) and its weights are applied unchanged to every layer, with
per-layer standardisation only. Refit-per-layer numbers are reported alongside.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(path):
    z = np.load(path, allow_pickle=False)
    H, key, cond = z["H"].astype(np.float32), z["key"], z["cond"]
    sid = np.array([k.split("|")[0] for k in key])
    return H, sid, cond


def dataset(H, sid, cond, pos="sufficient|none", neg="absent|none"):
    keep = np.isin(cond, [pos, neg])
    y = (cond[keep] == pos).astype(int)
    g = sid[keep]
    return H[keep], y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/lens_261.npz")
    ap.add_argument("--out", default="results/layer_transfer.json")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    H, sid, cond = load(args.run)
    X, y, g = dataset(H, sid, cond)
    n_layers = X.shape[1]
    oof_fixed = np.zeros((n_layers, len(y)))
    oof_refit = np.zeros((n_layers, len(y)))

    for tr, te in GroupKFold(n_splits=args.folds).split(X[:, 0], y, g):
        Xtr, Xte = X[tr], X[te]
        # single estimator on the final layer
        fin = make_pipeline(StandardScaler(),
                            LogisticRegression(C=args.C, max_iter=3000))
        fin.fit(Xtr[:, -1], y[tr])
        coef = fin[-1].coef_[0]
        b = fin[-1].intercept_[0]
        for L in range(n_layers):
            mu = Xtr[:, L].mean(0)
            sd = np.maximum(Xtr[:, L].std(0), 1e-6)
            oof_fixed[L, te] = ((Xte[:, L] - mu) / sd) @ coef + b
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=args.C, max_iter=3000))
            clf.fit(Xtr[:, L], y[tr])
            oof_refit[L, te] = clf.decision_function(Xte[:, L])

    res = {"n": int(len(y)), "layers": n_layers, "C": args.C,
           "fixed": [float(roc_auc_score(y, s)) for s in oof_fixed],
           "refit": [float(roc_auc_score(y, s)) for s in oof_refit]}
    best = int(np.argmax(res["fixed"]))
    res["best_fixed_layer"] = best
    res["best_fixed_auroc"] = res["fixed"][best]
    res["final_fixed_auroc"] = res["fixed"][-1]
    print(json.dumps({k: v for k, v in res.items() if k not in ("fixed", "refit")}, indent=2))
    print("fixed:", " ".join(f"{v:.3f}" for v in res["fixed"]))
    print("refit:", " ".join(f"{v:.3f}" for v in res["refit"]))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
