"""How much of the evidence signal is already in the answer distribution?

The probe is compared against every simple readout of the same 12-way softmax:
max prob, p(unknown), top-vs-unknown margin, entropy, and a learned linear
readout over the distribution. If the hidden-state probe only re-derives a
better readout of the logits, these close the gap.
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc
from .probe import PAIRS, grouped_ci, load


def signals(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    pc = p[:-1]                      # colours, last candidate is "unknown"
    pun = p[-1]
    nz = np.clip(p, 1e-12, 1)
    return {
        "maxprob": float(p.max()),
        "p_known": float(1.0 - pun),
        "margin": float(pc.max() - pun),
        "neg_entropy": float((p * np.log(nz)).sum()),
        "top2_gap": float(np.sort(pc)[-1] - np.sort(pc)[-2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v5_261.jsonl")
    ap.add_argument("--pairs", default="none,hint,correct")
    args = ap.parse_args()
    recs, names, H = load(args.run)

    for name in args.pairs.split(","):
        pos, neg = PAIRS[name]
        y, g, D, prob_vec = [], [], [], []
        for r in recs:
            for key, lab in ((pos, 1), (neg, 0)):
                p = r["probs"][key]
                y.append(lab)
                g.append(r["sample_id"])
                D.append(signals(p))
                prob_vec.append(p)
        y, g = np.array(y), np.array(g)
        prob_vec = np.array(prob_vec, dtype=np.float64)
        prob_vec /= prob_vec.sum(1, keepdims=True)
        print(f"\n== pair {name}: {pos} vs {neg}  n={len(y)}")

        for k in D[0]:
            s = np.array([d[k] for d in D])
            pt = auroc(s, y)
            lo, hi = grouped_ci(s, y, g)
            print(f"  {k:12s} AUROC {pt:.3f} [{lo:.3f}, {hi:.3f}]")

        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(prob_vec, y, g):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=3000))
            clf.fit(prob_vec[tr], y[tr])
            oof[te] = clf.decision_function(prob_vec[te])
        lo, hi = grouped_ci(oof, y, g)
        print(f"  {'logit_probe':12s} AUROC {auroc(oof, y):.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
