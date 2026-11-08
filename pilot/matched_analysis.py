"""Analyse the category-matched control runs.

For each model: frozen probe (fit on no-hint states of training categories,
C=0.1) scored on held-out categories under no hint and under conflict; refit
inside the conflict condition; and the mean-difference geometry between the
present and absent sides of the matched pairs.

Usage:
  python -m pilot.matched_analysis --run runs/matched_3b.jsonl --tag 3b
"""

import argparse
import json

import numpy as np

from .metrics import auroc


def load(run):
    rows = [json.loads(l) for l in open(run)]
    rows = [r for r in rows if "meta" not in r]
    z = np.load(run.replace(".jsonl", ".npz"))
    H = dict(zip(z["keys"].tolist(), z["H"]))
    return rows, H


def states(rows, H, key):
    if key == "none":
        return np.stack([H[r["sample_id"]] for r in rows]).astype(np.float32)
    return np.stack([H[f"{r['sample_id']}|{key}"] for r in rows]).astype(np.float32)


def fit_probe(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(X, y)
    return clf


def oof_scores(X, y, groups, C):
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    folds = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = fit_probe(X[tr], y[tr], C)
        oof[te] = clf.decision_function(X[te])
        folds.append((tr, te))
    return oof, folds


def frozen_scores(X_fit, y_fit, groups, X_eval, C):
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(X_eval))
    for tr, te in GroupKFold(n_splits=5).split(X_fit, y_fit, groups):
        clf = fit_probe(X_fit[tr], y_fit[tr], C)
        oof[te] = clf.decision_function(X_eval[te])
    return oof


def group_ci(score, y, groups, n_boot=2000, seed=0):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        vals.append(auroc(score[sel], y[sel]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def margin_ci(score, baseline, y, groups, n_boot=2000, seed=0):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        vals.append(auroc(score[sel], y[sel]) - auroc(baseline[sel], y[sel]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def geometry(Xn, Xc, y):
    m0p, m0n = Xn[y == 1].mean(0), Xn[y == 0].mean(0)
    mcp, mcn = Xc[y == 1].mean(0), Xc[y == 0].mean(0)
    d0, dc = m0p - m0n, mcp - mcn
    w = d0 / np.linalg.norm(d0)
    angle = float(np.degrees(np.arccos(np.clip(
        dc @ w / np.linalg.norm(dc), -1, 1))))
    return {
        "length_ratio": float(np.linalg.norm(dc) / np.linalg.norm(d0)),
        "angle_deg": angle,
        "along_ratio": float((dc @ w) / (d0 @ w)),
        "cos": float(np.clip(dc @ w / np.linalg.norm(dc), -1, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--group", default="category", choices=["category", "image"])
    ap.add_argument("--out", default="results/matched.json")
    args = ap.parse_args()

    rows, H = load(args.run)
    y = np.array([r["present"] for r in rows])
    if args.group == "image":
        groups = np.array([r["sample_id"].split("_")[1] for r in rows])
    else:
        groups = np.array([r["category"] for r in rows])
    Xn, Xc = states(rows, H, "none"), states(rows, H, "conflict")
    pk_n = np.array([1.0 - r["probs"]["none"][-1] for r in rows])
    pk_c = np.array([1.0 - r["probs"]["conflict"][-1] for r in rows])

    oof_n, _ = oof_scores(Xn, y, groups, args.C)
    frozen_c = frozen_scores(Xn, y, groups, Xc, args.C)
    oof_c, _ = oof_scores(Xc, y, groups, args.C)

    res = {"tag": args.tag, "n": int(len(y)), "pairs": int(len(y) // 2),
           "categories": int(len({r["category"] for r in rows})),
           "grouping": args.group,
           "groups": int(len(np.unique(groups))),
           "nohint": {"probe": auroc(oof_n, y),
                      "probe_ci": group_ci(oof_n, y, groups),
                      "p_unknown": auroc(pk_n, y)},
           "frozen_conflict": {"probe": auroc(frozen_c, y),
                               "probe_ci": group_ci(frozen_c, y, groups),
                               "p_unknown": auroc(pk_c, y)},
           "refit_conflict": {"probe": auroc(oof_c, y),
                              "probe_ci": group_ci(oof_c, y, groups)},
           "geometry": geometry(Xn, Xc, y)}
    res["frozen_minus_refit"] = res["frozen_conflict"]["probe"] - res["refit_conflict"]["probe"]
    res["frozen_minus_p_unknown"] = {
        "margin": res["frozen_conflict"]["probe"]
        - res["frozen_conflict"]["p_unknown"],
        "ci": margin_ci(frozen_c, pk_c, y, groups)}
    print(json.dumps(res, indent=1))
    try:
        old = json.load(open(args.out))
    except (FileNotFoundError, json.JSONDecodeError):
        old = {}
    old[args.tag] = res
    json.dump(old, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
