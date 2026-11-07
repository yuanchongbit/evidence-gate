"""Summarise the causal-control runs.

control_steer.jsonl: per-setting refusal/answer rates under conflict, next to
random directions orthogonal to w and dose-matched layer steering.

patch5.jsonl: refusal after patching same-image absent/present no-hint states
and their shuffled counterparts into the absent conflict pass, with recovery
relative to the no-hint/conflict span.
"""

import argparse
import json

import numpy as np

from .analyze_steer_conflict import METRICS, load_run, paired_diff, table


def label(probs, names):
    return names[int(np.argmax(probs))]


def boot_ratio(num, den, n_boot=2000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = len(num)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        vals.append(num[idx].mean() / d if abs(d) > 1e-9 else np.nan)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def summarise_steer(path, n_boot):
    recs, names = load_run(path)
    suffix = {key.rsplit("|", 1)[1] for key in recs[0]["probs"]
              if key.startswith("sufficient|wrong|")}
    settings = ["base"] + sorted([s for s in suffix if s != "base"],
                                 key=lambda s: (s[0] != "w", s))
    rows, means = table(recs, names, settings)
    result = {"n": len(recs), "means": means, "diff_vs_base": {
        s: paired_diff(rows, s, n_boot=n_boot) for s in settings if s != "base"}}

    small_w = [s for s in settings if s.startswith("w") and s.endswith("0.25")]
    small_r = [s for s in settings if s.startswith("r") and s.endswith("0.25")]
    if small_w and small_r:
        cmp = {}
        for m in METRICS:
            w_abs = np.mean([abs(result["diff_vs_base"][s][m]["diff"]) for s in small_w])
            r_abs = np.mean([abs(result["diff_vs_base"][s][m]["diff"]) for s in small_r])
            cmp[m] = {"w_abs_effect": float(w_abs),
                      "random_abs_effect": float(r_abs),
                      "ratio": float(w_abs / r_abs) if r_abs > 1e-9 else None}
        result["w_vs_random_at_0.25"] = cmp

    rand_abl = [s for s in settings if s.startswith("rabl")]
    if "ablate" in result["diff_vs_base"] and rand_abl:
        cmp = {}
        for m in METRICS:
            w_abs = abs(result["diff_vs_base"]["ablate"][m]["diff"])
            r_abs = np.mean([abs(result["diff_vs_base"][s][m]["diff"])
                             for s in rand_abl])
            cmp[m] = {"ablate_abs_effect": float(w_abs),
                      "random_ablate_abs_effect": float(r_abs),
                      "ratio": float(w_abs / r_abs) if r_abs > 1e-9 else None}
        result["ablate_vs_random_ablate"] = cmp
    return result


def patch_rows(recs, names):
    out = []
    base = np.array([float(label(r["probs"]["absent|a"], names) == "unknown")
                     for r in recs])
    none = np.array([float(label(r["probs"]["absent|none"], names) == "unknown")
                     for r in recs])
    conds = sorted({key.split("<-", 1)[1] for r in recs
                    for key in r["probs"] if "<-" in key})
    for cond in conds:
        key = f"absent|a<-{cond}"
        idx = np.array([i for i, r in enumerate(recs) if key in r["probs"]])
        ref = np.array([float(label(recs[i]["probs"][key], names) == "unknown")
                        for i in idx])
        hint = np.array([float(label(recs[i]["probs"][key], names)
                               == recs[i]["absent_hint"]) for i in idx])
        out.append((cond, ref, hint, idx))
    return base, none, dict((c, (r, h, i)) for c, r, h, i in out)


def summarise_patch(path, n_boot):
    lines = [json.loads(line) for line in open(path)]
    names = lines[0]["meta"]["candidates"].split(",")
    recs = [r for r in lines if "meta" not in r]
    base, none, conds = patch_rows(recs, names)

    means = {"base": {"refusal": float(base.mean()), "hint": None},
             "absent|none": {"refusal": float(none.mean()), "hint": None}}
    diff = {}
    recovery = {}
    for cond, (ref, hint, idx) in conds.items():
        b = base[idx]
        means[cond] = {"refusal": float(ref.mean()), "hint": float(hint.mean())}
        d = ref - b
        rng = np.random.default_rng(0)
        vals = [float(d[rng.integers(0, len(d), len(d))].mean())
                for _ in range(n_boot)]
        diff[cond] = {"refusal": float(d.mean()),
                      "ci": [float(np.quantile(vals, 0.025)),
                             float(np.quantile(vals, 0.975))]}
        recovery[cond] = {
            "point": float(d.mean() / (none[idx] - b).mean()),
            "ci": boot_ratio(d, none[idx] - b, n_boot=n_boot)}

    contrasts = {}
    for cond in conds:
        if not cond.startswith("same_abs@L"):
            continue
        L = cond.rsplit("L", 1)[1]
        other = f"same_pres@L{L}"
        shuf = f"shuf_abs@L{L}"
        if other in conds:
            d = conds[cond][0] - conds[other][0]
            rng = np.random.default_rng(0)
            vals = [float(d[rng.integers(0, len(d), len(d))].mean())
                    for _ in range(n_boot)]
            contrasts[f"same_abs-minus-same_pres@L{L}"] = {
                "diff": float(d.mean()),
                "ci": [float(np.quantile(vals, 0.025)),
                       float(np.quantile(vals, 0.975))]}
        if shuf in conds:
            a_idx, b_idx = conds[cond][2], conds[shuf][2]
            common = np.intersect1d(a_idx, b_idx)
            pa = {i: v for i, v in zip(a_idx, conds[cond][0])}
            pb = {i: v for i, v in zip(b_idx, conds[shuf][0])}
            d = np.array([pa[i] - pb[i] for i in common])
            rng = np.random.default_rng(0)
            vals = [float(d[rng.integers(0, len(d), len(d))].mean())
                    for _ in range(n_boot)]
            contrasts[f"same_abs-minus-shuf_abs@L{L}"] = {
                "diff": float(d.mean()),
                "ci": [float(np.quantile(vals, 0.025)),
                       float(np.quantile(vals, 0.975))]}

    return {"n": len(recs), "means": means, "diff_vs_base": diff,
            "recovery": recovery, "contrasts": contrasts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", default="runs/control_steer.jsonl")
    ap.add_argument("--patch", default="runs/patch5.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/control_causal.json")
    args = ap.parse_args()

    result = {}
    if args.steer:
        result["steer"] = summarise_steer(args.steer, args.n_boot)
    if args.patch:
        result["patch"] = summarise_patch(args.patch, args.n_boot)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
