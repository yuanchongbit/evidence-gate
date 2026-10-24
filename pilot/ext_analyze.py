"""Analyse the VizWiz external-validation run: behaviour, probe transfer under
hints, and risk-coverage against output signals."""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ext_vizwiz import is_refusal
from .metrics import auroc
from .probe import grouped_ci, grouped_diff, cv_oof, dataset, load as load_main

CONDS = ("none", "correct", "wrong")


def load(run):
    recs = [json.loads(l) for l in open(run)]
    z = np.load(run.replace(".jsonl", ".npz"), allow_pickle=False)
    H = dict(zip(z["keys"].tolist(), z["H"]))
    return recs, H


def stack(recs, H, cond):
    X, y, g = [], [], []
    for r in recs:
        if cond not in r["res"]:
            continue
        k = f"{r['sample_id']}|{cond}"
        X.append(H[k].astype(np.float32))
        y.append(int(r["answerable"]))
        g.append(r["sample_id"])
    return np.stack(X), np.array(y), np.array(g)


def cv_probe(recs, H, Cs):
    X, y, g = stack(recs, H, "none")
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    best = None
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.decision_function(X[te])
        pt = auroc(oof, y)
        if best is None or pt > best[0]:
            best = (pt, C)
    C = best[1]
    scores = {"none": np.full(len(recs), np.nan)}
    for c in CONDS[1:]:
        scores[c] = np.full(len(recs), np.nan)
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        te_ids = set(g[te])
        idx = [i for i, r in enumerate(recs) if r["sample_id"] in te_ids]
        for c in CONDS:
            sel = [i for i in idx if c in recs[i]["res"]]
            if not sel:
                continue
            Xc = np.stack([H[f"{recs[i]['sample_id']}|{c}"] for i in sel]).astype(np.float32)
            scores[c][sel] = clf.decision_function(Xc)
    return C, scores


def per_cond(recs):
    out = {}
    for c in CONDS:
        rows = [r["res"][c] for r in recs if c in r["res"]]
        out[c] = {
            "n": len(rows),
            "refusal": float(np.mean([r["refused"] for r in rows])),
            "p_unk": float(np.mean([r["p_unk"] for r in rows])),
            "maxprob": float(np.mean([r["maxprob"] for r in rows])),
            "acc": float(np.mean([r["acc"] for r in rows])),
        }
    return out


def rc(recs, score, covs=(1.0, 0.9, 0.7, 0.5)):
    """Selective risk: rank by score (higher = more willing to answer), take the
    top k = round(cov*n), risk = 1 - mean VQA accuracy among them."""
    s = np.array([score(i) for i in range(len(recs))], dtype=float)
    ok = ~np.isnan(s)
    s, idx = s[ok], np.where(ok)[0]
    acc = np.array([recs[i]["res"]["none"]["acc"] for i in idx])
    order = np.argsort(-s)
    rows = {}
    for cov in covs:
        k = max(1, int(round(cov * len(idx))))
        pick = order[:k]
        rows[f"{cov:.1f}"] = float(1.0 - acc[pick].mean())
    return rows


def transfer(main_run, recs, H, Cs):
    """Fit the probe on the 261-sample set and apply it unchanged to VizWiz."""
    mrecs, _, mH = load_main(main_run)
    X, y, g, _, _ = dataset(mrecs, mH, "sufficient|none", "absent|none", "sample")
    best = None
    for C in Cs:
        pt = auroc(cv_oof(X, y, g, C), y)
        if best is None or pt > best[0]:
            best = (pt, C)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000))
    clf.fit(X, y)
    print(f"\ntransfer probe: fitted on {len(y)} 261-set pairs, "
          f"in-domain OOF AUROC {best[0]:.3f} (C={best[1]:g})")
    out = {}
    for c in ("none", "wrong"):
        sel = [i for i, r in enumerate(recs) if c in r["res"]]
        Xw = np.stack([H[f"{recs[i]['sample_id']}|{c}"] for i in sel]).astype(np.float32)
        s = clf.decision_function(Xw)
        yy = np.array([int(recs[i]["answerable"]) for i in sel])
        gg = np.array([recs[i]["sample_id"] for i in sel])
        m = np.array([recs[i]["res"][c]["maxprob"] for i in sel])
        u = np.array([1.0 - recs[i]["res"][c]["p_unk"] for i in sel])
        lo, hi = grouped_ci(s, yy, gg)
        dlo, dhi = grouped_diff(s, u, yy, gg)
        print(f"  [{c}] frozen probe AUROC {auroc(s, yy):.3f} [{lo:.3f},{hi:.3f}]  "
              f"maxprob {auroc(m, yy):.3f}  p_known {auroc(u, yy):.3f}  "
              f"probe-p_known {auroc(s, yy)-auroc(u, yy):+.3f} [{dlo:+.3f},{dhi:+.3f}]")
        out[c] = {"auroc": auroc(s, yy), "lo": lo, "hi": hi,
                  "maxprob": auroc(m, yy), "p_known": auroc(u, yy), "n": len(sel)}
    return {"C": best[1], "in_domain_auroc": best[0], "by_cond": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ext_vizwiz.jsonl")
    ap.add_argument("--main-run", default="runs/v5_261.jsonl")
    ap.add_argument("--Cs", default="0.0003,0.001,0.003,0.01,0.1")
    args = ap.parse_args()
    recs, H = load(args.run)
    bad = [r for r in recs if r["answerable"] == 1 and is_refusal(r["gold"])]
    if bad:
        recs = [r for r in recs if not (r["answerable"] == 1 and is_refusal(r["gold"]))]
        print(f"dropped {len(bad)} answerable rows whose majority gold is a refusal")
    n_ans = sum(r["answerable"] for r in recs)
    print(f"{len(recs)} samples ({n_ans} answerable, {len(recs)-n_ans} unanswerable)")

    for arm, name in ((1, "answerable"), (0, "unanswerable")):
        rows = [r for r in recs if r["answerable"] == arm]
        print(f"\n== {name} (n={len(rows)})")
        for c in CONDS:
            rr = [r["res"][c] for r in rows if c in r["res"]]
            if not rr:
                continue
            print(f"  {c:8s} refuse {np.mean([x['refused'] for x in rr]):.3f} "
                  f"acc {np.mean([x['acc'] for x in rr]):.3f} "
                  f"p_unk {np.mean([x['p_unk'] for x in rr]):.3f} "
                  f"maxp {np.mean([x['maxprob'] for x in rr]):.3f}")

    C, probe = cv_probe(recs, H, [float(c) for c in args.Cs.split(",")])
    print(f"\nprobe C={C:g}")
    for c in ("none", "wrong"):
        _, y, g = stack(recs, H, c)
        p = probe[c]
        m = np.array([r["res"][c]["maxprob"] for r in recs if c in r["res"]])
        u = np.array([1.0 - r["res"][c]["p_unk"] for r in recs if c in r["res"]])
        print(f"  [{c}] probe AUROC {auroc(p, y):.3f} "
              f"({grouped_ci(p, y, g)[0]:.3f},{grouped_ci(p, y, g)[1]:.3f})  "
              f"maxprob {auroc(m, y):.3f} "
              f"(d {auroc(p,y)-auroc(m,y):+.3f} "
              f"[{grouped_diff(p, m, y, g)[0]:+.3f},{grouped_diff(p, m, y, g)[1]:+.3f}])  "
              f"p_known {auroc(u, y):.3f} "
              f"(d {auroc(p,y)-auroc(u,y):+.3f} "
              f"[{grouped_diff(p, u, y, g)[0]:+.3f},{grouped_diff(p, u, y, g)[1]:+.3f}])")

    print("\nrisk (1-acc) at coverage, ranked by signal (all samples, no-hint)")
    print("  signal   " + "  ".join(f"cov={c}" for c in ("1.0", "0.9", "0.7", "0.5")))
    sig = {
        "probe": lambda i: probe["none"][i],
        "maxprob": lambda i: recs[i]["res"]["none"]["maxprob"],
        "p_known": lambda i: 1.0 - recs[i]["res"]["none"]["p_unk"],
        "refuse": lambda i: 0.0 if recs[i]["res"]["none"]["refused"] else 1.0,
    }
    out = {}
    for k, fn in sig.items():
        r = rc(recs, fn)
        out[k] = r
        print(f"  {k:8s} " + "  ".join(f"{r[c]:.3f}" for c in ("1.0", "0.9", "0.7", "0.5")))

    tr = transfer(args.main_run, recs, H, [float(c) for c in args.Cs.split(",")])

    summary = {
        "n": len(recs), "n_answerable": int(n_ans), "C": C,
        "by_arm": {name: per_cond([r for r in recs if r["answerable"] == a])
                   for a, name in ((1, "answerable"), (0, "unanswerable"))},
        "risk": out, "transfer": tr,
    }
    with open(args.run.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved", args.run.replace(".jsonl", "_summary.json"))


if __name__ == "__main__":
    main()
