"""Summarise the conflict steering run: refusal and answer shifts per setting,
with paired cluster-bootstrap CIs over scenes."""

import argparse
import json

import numpy as np

from .constants import DISTRACTOR

METRICS = ("present_refusal", "present_gold", "present_hint",
           "absent_refusal", "absent_hint")


def load_run(path):
    lines = [json.loads(line) for line in open(path)]
    names = lines[0]["meta"]["candidates"].split(",")
    return [rec for rec in lines if "meta" not in rec], names


def label(rec, key, names):
    return names[int(np.argmax(rec["probs"][key]))]


def scene_metrics(rec, names, setting):
    present = label(rec, f"sufficient|wrong|{setting}", names)
    hint_p = DISTRACTOR[rec["gold"]]
    absent = [label(rec, f"absent|a|{setting}", names),
              label(rec, f"absent|b|{setting}", names)]
    hints_a = [rec["absent_hint"], DISTRACTOR[rec["absent_hint"]]]
    return {
        "present_refusal": float(present == "unknown"),
        "present_gold": float(present == rec["gold"]),
        "present_hint": float(present == hint_p),
        "absent_refusal": float(np.mean([a == "unknown" for a in absent])),
        "absent_hint": float(np.mean([a == h for a, h in zip(absent, hints_a)])),
    }


def table(recs, names, settings):
    rows = {setting: [scene_metrics(rec, names, setting) for rec in recs]
            for setting in settings}
    out = {}
    for setting in settings:
        out[setting] = {m: float(np.mean([r[m] for r in rows[setting]]))
                        for m in METRICS}
    return rows, out


def paired_diff(rows, setting, base="base", n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rows[base])
    out = {}
    for metric in METRICS:
        diff = np.array([a[metric] - b[metric]
                         for a, b in zip(rows[setting], rows[base])])
        vals = [float(np.mean(diff[rng.integers(0, n, n)]))
                for _ in range(n_boot)]
        out[metric] = {"diff": float(diff.mean()),
                       "ci": [float(np.quantile(vals, 0.025)),
                              float(np.quantile(vals, 0.975))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/steer_conflict.jsonl")
    ap.add_argument("--baseline", default="runs/v5_261.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/steer_conflict.json")
    args = ap.parse_args()

    recs, names = load_run(args.run)
    suffixes = {key.rsplit("|", 1)[1] for key in recs[0]["probs"]
                if key.startswith("sufficient|wrong|")}
    settings = ["base", "ablate"] + sorted(
        [s for s in suffixes if s not in ("base", "ablate")],
        key=lambda s: float(s[1:]))

    rows, means = table(recs, names, settings)
    result = {"n": len(recs), "means": means, "diff_vs_base": {}}
    for setting in settings:
        if setting != "base":
            result["diff_vs_base"][setting] = paired_diff(
                rows, setting, n_boot=args.n_boot)

    if args.baseline:
        base_recs, base_names = load_run(args.baseline)
        by_id = {rec["sample_id"]: rec for rec in base_recs}
        vals = {m: [] for m in METRICS}
        for rec in recs:
            b = by_id[rec["sample_id"]]
            present = b["probs"]["sufficient|none"]
            plabel = base_names[int(np.argmax(present))]
            absent = base_names[int(np.argmax(b["probs"]["absent|none"]))]
            vals["present_refusal"].append(float(plabel == "unknown"))
            vals["present_gold"].append(float(plabel == b["gold"]))
            vals["present_hint"].append(None)
            vals["absent_refusal"].append(float(absent == "unknown"))
            vals["absent_hint"].append(None)
        result["no_hint"] = {m: (None if v[0] is None else float(np.mean(v)))
                             for m, v in vals.items()}

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
