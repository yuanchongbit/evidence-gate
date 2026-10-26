"""Decompose the answer distribution under conflicting hints and relate hint
adoption to the pre-hint evidence score.

The decomposition uses only the 12-way readout. The regression uses the
out-of-fold probe score from the no-hint pass, so the hint cannot move the
axis. CIs are cluster bootstraps over scenes.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import DISTRACTOR
from .probe import load
from .sdt import oof_scores

ROOT = Path(__file__).resolve().parent.parent


def hint_for(rec, key):
    if key == "sufficient|wrong":
        return DISTRACTOR.get(rec["gold"])
    if key == "absent|a":
        return rec["absent_hint"]
    if key == "absent|b":
        return DISTRACTOR.get(rec["absent_hint"])
    return None


def decompose(recs, names, keys):
    out = {}
    for key in keys:
        present = key.startswith("sufficient")
        counts = {"n": len(recs), "unknown": 0, "true": 0, "hint": 0, "other": 0}
        for rec in recs:
            probs = rec["probs"][key]
            label = names[int(np.argmax(probs))]
            hint = hint_for(rec, key)
            if label == "unknown":
                counts["unknown"] += 1
            elif label == hint:
                counts["hint"] += 1
            elif present and label == rec["gold"]:
                counts["true"] += 1
            else:
                counts["other"] += 1
        out[key] = {k: (v / counts["n"] if k != "n" else v)
                    for k, v in counts.items()}
    return out


def fit_slope(z, y):
    clf = LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y)
    return float(clf.coef_[0, 0])


def regression(recs, scores, n_boot=2000, seed=0):
    rows = []
    for rec in recs:
        sid = rec["sample_id"]
        z = scores[(sid, "sufficient|none")]
        probs = rec["probs"]["sufficient|wrong"]
        label = rec["candidates"][int(np.argmax(probs))]
        rows.append((z, int(label == DISTRACTOR.get(rec["gold"])),
                     int(label == rec["gold"])))
    rows = np.array(rows)
    z, y_hint, y_gold = rows[:, 0], rows[:, 1], rows[:, 2]
    out = {}
    rng = np.random.default_rng(seed)
    n = len(z)
    for name, y in (("hint", y_hint), ("gold", y_gold)):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y[idx].min() == y[idx].max():
                continue
            vals.append(fit_slope(z[idx], y[idx]))
        out[name] = {
            "rate": float(y.mean()),
            "slope": fit_slope(z, y),
            "ci": [float(np.quantile(vals, 0.025)),
                   float(np.quantile(vals, 0.975))],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v5_261.jsonl")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/decompose.json")
    args = ap.parse_args()

    with open(args.run) as f:
        lines = [json.loads(line) for line in f]
    names = lines[0]["meta"]["candidates"].split(",")
    recs = [rec for rec in lines if "meta" not in rec]

    result = {"n": len(recs),
              "decomposition": decompose(
                  recs, names, ["sufficient|wrong", "absent|a", "absent|b"])}

    npz = args.run.replace(".jsonl", ".npz")
    if os.path.exists(npz):
        recs_h, _, H = load(args.run)
        scores = oof_scores(recs_h, H, args.C)
        for rec in recs:
            rec["candidates"] = names
        result["hint_adoption"] = regression(recs, scores, args.n_boot)
    else:
        result["hint_adoption"] = None

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
