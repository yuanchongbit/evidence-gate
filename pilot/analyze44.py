"""Summarise the v2 control batch (extra44, patch6, prompt variants, presence,
resolution) into one JSON per model."""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import DISTRACTOR
from .metrics import auroc
from .probe import cv_oof, load


def load_jsonl(path):
    lines = [json.loads(line) for line in open(path)]
    return lines[0].get("meta"), [r for r in lines if "meta" not in r]


def lab(p, names):
    return names[int(np.argmax(p))]


def boot_diff(a, b, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    vals = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)]
    return float(d.mean()), [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def conflict_metrics(recs, names, cond="absent|a"):
    ref = [float(lab(r["probs"][cond], names) == "unknown") for r in recs]
    hint = [float(lab(r["probs"][cond], names) == r["absent_hint"]) for r in recs]
    return ref, hint


def split_key(k):
    parts = k.split("|")
    return "|".join(parts[:2]), ("|".join(parts[2:]) if len(parts) > 2 else "base")


def extra_summary(path):
    meta, recs = load_jsonl(path)
    names = meta["candidates"].split(",")
    settings = sorted({split_key(k)[1] for k in recs[0]["probs"]})

    def rows_for(s):
        return [{**r, "probs": {split_key(k)[0]: v for k, v in r["probs"].items()
                                if split_key(k)[1] == s}} for r in recs]

    def ab_rates(rows):
        out = []
        for key, hinted in (("absent|a", "a"), ("absent|b", "b")):
            ref, hint = [], []
            for r in rows:
                answer = lab(r["probs"][key], names)
                target = (r["absent_hint"] if hinted == "a"
                          else DISTRACTOR[r["absent_hint"]])
                ref.append(float(answer == "unknown"))
                hint.append(float(answer == target))
            out.append((np.array(ref), np.array(hint)))
        return (out[0][0] + out[1][0]) / 2, (out[0][1] + out[1][1]) / 2

    base_rows = rows_for("base")
    base_ref, base_hint = conflict_metrics(base_rows, names)
    base_ref_ab, base_hint_ab = ab_rates(base_rows)
    acc = lambda rs: [float(lab(r["probs"]["sufficient|wrong"], names) == r["gold"])
                      for r in rs if "sufficient|wrong" in r["probs"]]
    out = {"n": len(recs), "energy_frac": meta.get("energies"),
           "removal_k": meta.get("removal_k"), "settings": {}}
    for s in settings:
        rows = rows_for(s)
        entry = {}
        if all("absent|a" in r["probs"] for r in rows):
            ref, hint = conflict_metrics(rows, names)
            entry["refusal"] = float(np.mean(ref))
            entry["hint"] = float(np.mean(hint))
            if s != "base":
                entry["d_refusal_ci"] = boot_diff(ref, base_ref)
                entry["d_hint_ci"] = boot_diff(hint, base_hint)
            if all("absent|b" in r["probs"] for r in rows):
                ref_ab, hint_ab = ab_rates(rows)
                entry["refusal_ab"] = float(np.mean(ref_ab))
                entry["hint_ab"] = float(np.mean(hint_ab))
                if s != "base":
                    entry["d_refusal_ab_ci"] = boot_diff(ref_ab, base_ref_ab)
                    entry["d_hint_ab_ci"] = boot_diff(hint_ab, base_hint_ab)
        a = acc(rows)
        if a:
            entry["acc_wrong"] = float(np.mean(a))
        out["settings"][s] = entry
    return out


def patch6_summary(path):
    meta, recs = load_jsonl(path)
    names = meta["candidates"].split(",")
    out = {"n": len(recs), "layers": {}}
    for L in meta.get("noop_layers", meta["layers"]):
        same_noop, same_zero, dmax = [], [], []
        ref_change = []
        for r in recs:
            b = r["probs"]["absent|a"]
            n = r["probs"][f"absent|a|noop@L{L}"]
            z = r["probs"][f"absent|a|zero@L{L}"]
            same_noop.append(float(np.argmax(b) == np.argmax(n)))
            same_zero.append(float(np.argmax(b) == np.argmax(z)))
            dmax.append(float(np.max(np.abs(np.array(b) - np.array(n)))))
            ref_change.append(float((lab(n, names) == "unknown")
                                    - (lab(b, names) == "unknown")))
        out["layers"][str(L)] = {
            "argmax_equal_noop": float(np.mean(same_noop)),
            "max_abs_prob_diff_noop": float(np.max(dmax)),
            "argmax_equal_zero": float(np.mean(same_zero)),
            "refusal_change_zero": float(np.mean(ref_change))}
    return out


def prompt_summary(path):
    meta, recs = load_jsonl(path)
    names = meta["candidates"].split(",")
    out = {"n": len(recs), "formats": {}}
    for fmt in ("std", "nounk", "nl"):
        e = {}
        for cond in ("absent|none", "absent|a", "absent|b"):
            key = f"{fmt}|{cond}"
            target = [(DISTRACTOR[r["absent_hint"]] if cond == "absent|b"
                       else r["absent_hint"]) for r in recs]
            ref = [float(lab(r["probs"][key], names) == "unknown") for r in recs]
            hint = [float(lab(r["probs"][key], names) == t)
                    for r, t in zip(recs, target)]
            e[cond] = {"refusal": float(np.mean(ref)), "hint": float(np.mean(hint))}
        acc = [float(lab(r["probs"][f"{fmt}|sufficient|wrong"], names) == r["gold"])
               for r in recs]
        e["acc_wrong"] = float(np.mean(acc))
        out["formats"][fmt] = e
    return out


def presence_summary(path, native_run):
    p = [json.loads(l) for l in open(path)]
    p_yes = np.array([r["p_yes"] for r in p])
    out = {"n": len(p), "mean_p_yes": float(p_yes.mean()),
           "frac_p_yes_gt_0.5": float((p_yes > 0.5).mean()),
           "frac_p_yes_gt_0.2": float((p_yes > 0.2).mean())}
    if native_run:
        recs, names, H = load(native_run)
        keep = {r["sample_id"] for r in p if r["p_yes"] < 0.5}
        sub = [r for r in recs if r["sample_id"] in keep]
        if sub:
            ref = [float(lab(r["probs"]["absent|a"], names) == "unknown") for r in sub]
            refn = [float(lab(r["probs"]["absent|none"], names) == "unknown") for r in sub]
            hint = [float(lab(r["probs"]["absent|a"], names) == r["absent_hint"])
                    for r in sub]
            out["strict"] = {"n": len(sub), "refusal_hint": float(np.mean(ref)),
                             "refusal_none": float(np.mean(refn)),
                             "hint": float(np.mean(hint))}
            full = [r for r in recs if r.get("absent_category")]
            ref_all = [float(lab(r["probs"]["absent|a"], names) == "unknown") for r in full]
            out["all"] = {"n": len(full), "refusal_hint": float(np.mean(ref_all))}
            Xl, yl, gl = [], [], []
            for r in sub:
                for key, yv in (("sufficient|wrong", 1), ("absent|a", 0)):
                    k = f"{r['sample_id']}|{key}"
                    if k in H:
                        Xl.append(H[k].astype(np.float32))
                        yl.append(yv)
                        gl.append(r["sample_id"])
            if len(set(yl)) == 2:
                Xl, yl, gl = np.stack(Xl), np.array(yl), np.array(gl)
                oof = cv_oof(Xl, yl, gl, 0.1)
                out["strict"]["probe_auroc"] = float(auroc(oof, yl))
    return out


def resolution_summary(path, native_run):
    meta, recs = load_jsonl(path)
    names = meta["candidates"].split(",")

    def m(rs):
        return {"refusal_none": float(np.mean([lab(r["probs"]["absent|none"], names) == "unknown" for r in rs])),
                "refusal_a": float(np.mean([lab(r["probs"]["absent|a"], names) == "unknown" for r in rs])),
                "hint_a": float(np.mean([lab(r["probs"]["absent|a"], names) == r["absent_hint"] for r in rs])),
                "acc_wrong": float(np.mean([lab(r["probs"]["sufficient|wrong"], names) == r["gold"] for r in rs]))}
    out = {"n": len(recs), "max_pixels": meta.get("max_pixels"), "metrics": m(recs)}
    # frozen probe fitted on the native battery, applied here
    try:
        base_recs, _, _ = load(native_run)
        z = np.load(native_run.replace(".jsonl", ".npz"), allow_pickle=False)
        Hn = dict(zip(z["keys"].tolist(), z["H"]))
        Xs = np.stack([Hn[f"{r['sample_id']}|sufficient|none"].astype(np.float32)
                       for r in base_recs if f"{r['sample_id']}|sufficient|none" in Hn])
        Xa = np.stack([Hn[f"{r['sample_id']}|absent|none"].astype(np.float32)
                       for r in base_recs if f"{r['sample_id']}|absent|none" in Hn])
        X = np.vstack([Xs, Xa]); y = np.r_[np.ones(len(Xs)), np.zeros(len(Xa))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(X, y)
        z2 = np.load(path.replace(".jsonl", ".npz"), allow_pickle=False)
        H = dict(zip(z2["keys"].tolist(), z2["H"]))
        keys = [f"{r['sample_id']}|sufficient|none" for r in recs] + \
               [f"{r['sample_id']}|absent|none" for r in recs]
        Xt = np.stack([H[k].astype(np.float32) for k in keys if k in H])
        yt = np.r_[np.ones(len(recs)), np.zeros(len(recs))][:len(Xt)]
        out["frozen_probe_auroc"] = float(auroc(clf.decision_function(Xt), yt))
    except Exception as e:  # noqa: BLE001
        out["frozen_probe_error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra")
    ap.add_argument("--patch6")
    ap.add_argument("--prompt")
    ap.add_argument("--presence")
    ap.add_argument("--resolution")
    ap.add_argument("--native-run", default="runs/v5_261.jsonl")
    ap.add_argument("--out", default="results/batch44.json")
    args = ap.parse_args()

    out = {}
    if args.extra:
        out["extra"] = extra_summary(args.extra)
    if args.patch6:
        out["patch6"] = patch6_summary(args.patch6)
    if args.prompt:
        out["prompt"] = prompt_summary(args.prompt)
    if args.presence:
        out["presence"] = presence_summary(args.presence, args.native_run)
    if args.resolution:
        out["resolution"] = resolution_summary(args.resolution, args.native_run)
    print(json.dumps(out, indent=2)[:8000])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
