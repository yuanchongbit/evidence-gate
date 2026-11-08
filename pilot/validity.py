"""Construct and predictive validity checks for the evidence probe.

Two questions the reviews raised:

1. Is the probe tracking evidence availability or merely answerability?
   Restrict the present arm to items the model answers correctly, and to items
   it answers wrongly (both are answerable), and compare each with the absent
   arm. A probe that tracks answerability should lose its separation on the
   answered-wrong subset; one that tracks presence should not.

2. Does the representation predict behaviour? For each scene, take the frozen
   probe's present-minus-absent separation in a condition and ask whether it
   ranks the model's own outcome (refuses the absent item, answers the present
   item, adopts the hint) in that same condition.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc
from .probe import load

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "3b": "results/runs/v5_261.jsonl",
    "7b": "results/runs/v5_261_7b.jsonl",
    "llava": "results/runs/v5_261_llava.jsonl",
    "gemma": "results/runs/v5_261_gemma.jsonl",
    "phi": "results/runs/phi_261.jsonl",
}


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000))


def oof_scores(H, recs, pos, neg):
    X, y, g, keys = [], [], [], []
    for r in recs:
        for k, lab in ((pos, 1), (neg, 0)):
            X.append(H[f"{r['sample_id']}|{k}"].astype(np.float32))
            y.append(lab)
            g.append(r["sample_id"])
            keys.append(f"{r['sample_id']}|{k}")
    X, y, g = np.stack(X), np.array(y), np.array(g)
    out = {}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = make_clf().fit(X[tr], y[tr])
        for i in te:
            out[keys[i]] = float(clf.decision_function(X[i:i + 1])[0])
    return out


def fit_full(H, recs, pos, neg):
    X, y = [], []
    for r in recs:
        for k, lab in ((pos, 1), (neg, 0)):
            X.append(H[f"{r['sample_id']}|{k}"].astype(np.float32))
            y.append(lab)
    return make_clf().fit(np.stack(X), np.array(y))


def boot_ci(scores, y, groups, n_boot=2000, seed=0):
    uniq = np.unique(groups)
    idx_of = {k: np.where(groups == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(scores[idx], y[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def subset_check(recs, names, H):
    score = oof_scores(H, recs, "sufficient|none", "absent|none")
    correct = {}
    for r in recs:
        p = r["probs"]["sufficient|none"]
        correct[r["sample_id"]] = names[int(np.argmax(p))] == r["gold"]
    out = {}
    for label, keep_pos in (("all_present", lambda sid: True),
                            ("present_correct", lambda sid: correct[sid]),
                            ("present_wrong", lambda sid: not correct[sid])):
        s, y, g = [], [], []
        for r in recs:
            sid = r["sample_id"]
            if not keep_pos(sid):
                continue
            s.append(score[f"{sid}|sufficient|none"])
            y.append(1)
            g.append(sid)
            s.append(score[f"{sid}|absent|none"])
            y.append(0)
            g.append(sid)
        s, y, g = np.array(s), np.array(y), np.array(g)
        out[label] = {"n": int(len(y)), "auroc": float(auroc(s, y)),
                      "ci": boot_ci(s, y, g)}
    s, y, g = [], [], []
    for r in recs:
        sid = r["sample_id"]
        s.append(score[f"{sid}|sufficient|none"])
        y.append(int(correct[sid]))
        g.append(sid)
    out["within_present_correct_vs_wrong"] = {
        "n": int(len(y)), "auroc": float(auroc(np.array(s), np.array(y))),
        "ci": boot_ci(np.array(s), np.array(y), np.array(g))}
    return out


def behaviour_check(recs, names, H):
    # frozen no-hint probe, cross-fitted conflict scores
    score_c = oof_scores(H, recs, "sufficient|wrong", "absent|a")
    score_n = oof_scores(H, recs, "sufficient|none", "absent|none")
    out = {}
    for cond, pos, neg, zmap in (
            ("none", "sufficient|none", "absent|none", score_n),
            ("conflict", "sufficient|wrong", "absent|a", score_c)):
        sep, gold, adopt, refuse = [], [], [], []
        for r in recs:
            sid = r["sample_id"]
            sep.append(zmap[f"{sid}|{pos}"] - zmap[f"{sid}|{neg}"])
            lab_p = names[int(np.argmax(r["probs"][pos]))]
            lab_a = names[int(np.argmax(r["probs"][neg]))]
            gold.append(int(lab_p == r["gold"]))
            adopt.append(int(lab_p != r["gold"] and lab_p != "unknown"))
            refuse.append(int(lab_a == "unknown"))
        sep = np.array(sep)
        rows = {}
        for name, y in (("present_gold", gold), ("present_adopt", adopt),
                        ("absent_refuse", refuse)):
            y = np.array(y)
            rho = spearmanr(sep, y)
            rows[name] = {
                "rate": float(y.mean()),
                "rho": float(rho.correlation),
                "p": float(rho.pvalue),
                "auroc": float(auroc(sep, y)),
                "ci": boot_ci(sep, y, np.array([r["sample_id"] for r in recs])),
            }
        out[cond] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validity.json")
    args = ap.parse_args()
    res = {}
    for tag, path in MODELS.items():
        recs, names, H = load(str(ROOT / path))
        res[tag] = {"subset": subset_check(recs, names, H),
                    "behaviour": behaviour_check(recs, names, H)}
        print(tag, json.dumps(res[tag], indent=1)[:400], flush=True)
    with open(ROOT / args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
