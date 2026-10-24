"""Analysis for the v5 design: present object vs verified-absent object.

Evidence insufficiency is operationalised without image editing: each image is
paired with a category that COCO annotations place outside the scene, so the
"absent" case has no removal artifact to exploit.
"""

import argparse
import json
import statistics

import numpy as np

from .constants import CANDIDATES, DISTRACTOR
from .metrics import auroc, aurc, accuracy_at_coverage, ece, hellinger

mean = statistics.mean
PRESENT_HINTS = ["correct", "wrong", "none"]
ABSENT_HINTS = ["a", "b", "none"]


def load(path):
    lines = [json.loads(l) for l in open(path)]
    meta = lines[0]["meta"]
    recs = [r for r in lines if "meta" not in r]
    return meta, recs, meta["candidates"].split(",")


def pred(p):
    i = int(np.argmax(p))
    return i, float(p[i])


def condition_stats(recs, names):
    out = {}
    for key in [f"sufficient|{h}" for h in PRESENT_HINTS] + [f"absent|{h}" for h in ABSENT_HINTS]:
        labels, confs, refused, hinted, correct = [], [], [], [], []
        for r in recs:
            p = r["probs"][key]
            i, c = pred(p)
            lab = names[i]
            labels.append(lab)
            confs.append(c)
            refused.append(lab == "unknown")
            if key.startswith("sufficient"):
                hinted.append(key.endswith("|wrong") and lab == DISTRACTOR[r["gold"]])
                correct.append(lab == r["gold"])
            else:
                h = r["absent_hint"] if key.endswith("|a") else DISTRACTOR[r["absent_hint"]]
                hinted.append(lab == h)
                correct.append(False)
        out[key] = {
            "n": len(labels),
            "refuse": mean(refused),
            "conf": mean(confs),
            "hinted": mean(hinted) if not key.endswith("|none") else float("nan"),
            "acc": mean(correct) if key.startswith("sufficient") else float("nan"),
            "top": {lab: labels.count(lab) for lab in sorted(set(labels))},
        }
    return out


def edg_conf(recs, names):
    s, y = [], []
    for r in recs:
        for key, lab in (("sufficient|none", 1), ("absent|none", 0)):
            p = r["probs"][key]
            s.append(pred(p)[1])
            y.append(lab)
    s, y = np.array(s), np.array(y)
    point = auroc(s, y)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        vals.append(auroc(s[idx], y[idx]))
    return point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v5_261.jsonl")
    args = ap.parse_args()
    meta, recs, names = load(args.run)
    print(f"model {meta['model']}  n={len(recs)}")

    tab = condition_stats(recs, names)
    print(f"\n{'condition':<18}{'n':>4}{'refuse':>8}{'conf':>7}{'hinted':>8}{'acc':>7}")
    for k, v in tab.items():
        print(f"{k:<18}{v['n']:>4}{v['refuse']:>8.3f}{v['conf']:>7.3f}"
              f"{v['hinted']:>8.3f}{v['acc']:>7.3f}")

    uar = mean(tab[f"absent|{h}"]["refuse"] for h in ABSENT_HINTS)
    oar = mean(tab[f"sufficient|{h}"]["refuse"] for h in PRESENT_HINTS)
    print(f"\nUAR (answers when object absent) {1-uar:.3f}")
    print(f"OAR (refuses when object present) {oar:.3f}")
    print(f"hint compliance present|wrong {tab['sufficient|wrong']['hinted']:.3f}")
    print(f"hint compliance absent|a {tab['absent|a']['hinted']:.3f}")
    print(f"hint compliance absent|b {tab['absent|b']['hinted']:.3f}")

    pt, lo, hi = edg_conf(recs, names)
    print(f"\nEDG[conf] {pt:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\nabsent|none label distribution:", tab["absent|none"]["top"])
    print("present|none label distribution:", tab["sufficient|none"]["top"])

    confs = [pred(r["probs"]["sufficient|none"])[1] for r in recs]
    corr = [names[pred(r["probs"]["sufficient|none"])[0]] == r["gold"] for r in recs]
    print(f"ECE present|none {ece(confs, corr):.3f}")
    print(f"AURC present|none {aurc(confs, corr):.3f}  acc@80 "
          f"{accuracy_at_coverage(confs, corr, 0.8):.3f}")


if __name__ == "__main__":
    main()
