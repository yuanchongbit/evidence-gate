"""Behaviour summary for the alternative task forms (task2 side, task3 count)."""

import argparse
import json
from collections import Counter

import numpy as np

from .probe import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hint-answer", default=None,
                    help="candidate asserted by the hint; inferred from task if omitted")
    args = ap.parse_args()
    recs, names, _ = load(args.run)
    unk = "unknown"
    hint_val = args.hint_answer
    if hint_val is None:
        hint_val = "two" if "two" in names else None

    print(f"n={len(recs)}  candidates={names}")
    for key in recs[0]["probs"]:
        p = np.array([r["probs"][key] for r in recs])
        top = [names[i] for i in p.argmax(1)]
        cnt = Counter(top)
        ref = float(cnt[unk] / len(top))
        adopt = float(cnt[hint_val] / len(top)) if hint_val else float("nan")
        if "none" in key:
            acc = np.mean([names[i] == r["gold"] for i, r in zip(p.argmax(1), recs)])
        else:
            acc = float("nan")
        share = " ".join(f"{k}:{v/len(top):.2f}" for k, v in cnt.most_common())
        print(f"  {key:18s} refuse {ref:.3f}  adopt_hint {adopt:.3f}  "
              f"acc {acc:.3f}  top1 {share}")


if __name__ == "__main__":
    main()
