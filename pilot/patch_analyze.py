"""Summarise runs/patch.jsonl: mean p(unknown) and paired deltas vs the
unpatched target condition."""

import argparse
import json

import numpy as np

from .constants import CANDIDATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/patch.jsonl")
    args = ap.parse_args()
    lines = [json.loads(l) for l in open(args.out)]
    recs = [r for r in lines if "meta" not in r]
    if not recs:
        raise SystemExit(f"{args.out}: no records")
    keys = list(dict.fromkeys(k for r in recs for k in r["probs"]))
    unk = len(CANDIDATES) - 1
    print(f"n={len(recs)}")

    rng = np.random.default_rng(0)
    for k in keys:
        rows = [r for r in recs if k in r["probs"]]
        pk = np.array([r["probs"][k][unk] for r in rows])
        if "<-" in k:
            target = k.split("<-")[0]
            base = np.array([r["probs"][target][unk] for r in rows])
            d = pk - base
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            print(f"  {k:44s} p_unknown {pk.mean():.3f}  base {base.mean():.3f}  "
                  f"delta {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(rows)}")
        else:
            print(f"  {k:44s} p_unknown {pk.mean():.3f}  n={len(rows)}")


if __name__ == "__main__":
    main()
