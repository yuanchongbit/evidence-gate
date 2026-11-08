"""Fabrication at matched answered coverage on the category-matched control.

Same convention as pilot/deploy_check.gate_answered: select a prefix by score
until the model has answered the target fraction of all items, then report the
absent fraction among answered items. The probe is the frozen no-hint probe
(grouped transfer within the matched pairs).

Usage:
  python -m pilot.matched_gate --run runs/matched_3b.jsonl --tag 3b
"""

import argparse
import json

import numpy as np

from .matched_analysis import frozen_scores, load, states


def gate_answered(score, is_absent, answered, coverage):
    order = np.argsort(-score, kind="stable")
    target = int(round(coverage * len(score)))
    cum = np.cumsum(answered[order])
    k = int(np.searchsorted(cum, target, side="left") + 1)
    k = min(k, len(score))
    sel = np.zeros(len(score), bool)
    sel[order[:k]] = True
    ans = answered[sel]
    n = int(ans.sum())
    return {"answered_coverage": float(n / len(score)),
            "selection_coverage": float(k / len(score)),
            "fabrication_among_answered": float(
                (is_absent[sel] & ans).sum() / n) if n else None,
            "n_answered": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--group", default="category", choices=["category", "image"])
    ap.add_argument("--out", default="results/matched_gate.json")
    args = ap.parse_args()

    rows, H = load(args.run)
    y = np.array([r["present"] for r in rows])
    if args.group == "image":
        groups = np.array([r["sample_id"].split("_")[1] for r in rows])
    else:
        groups = np.array([r["category"] for r in rows])
    Xn, Xc = states(rows, H, "none"), states(rows, H, "conflict")
    probs = np.stack([r["probs"]["conflict"] for r in rows])
    answered = probs.argmax(1) != probs.shape[1] - 1

    frozen_c = frozen_scores(Xn, y, groups, Xc, args.C)
    scorers = {
        "probe": frozen_c,
        "p_known": 1.0 - probs[:, -1],
        "maxprob": probs[:, :-1].max(1),
    }
    res = {"tag": args.tag, "n": int(len(y)), "answered": float(answered.mean()),
           "grouping": args.group}
    for name, sc in scorers.items():
        res[name] = {f"{c:.2f}": gate_answered(sc, ~y.astype(bool), answered, c)
                     for c in (0.25, 0.50)}
    print(json.dumps(res, indent=1))
    try:
        old = json.load(open(args.out))
    except (FileNotFoundError, json.JSONDecodeError):
        old = {}
    old[args.tag] = res
    json.dump(old, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
