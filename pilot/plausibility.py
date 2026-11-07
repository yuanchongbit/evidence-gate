"""Is abstention driven by evidence absence or by an out-of-place category?

The absent category is sampled without constraining it to fit the scene, so a
model could refuse because a zebra in a kitchen is implausible rather than
because the zebra is missing. We score each absent category against the objects
actually annotated in that image, using COCO train co-occurrence,

    s = mean_c P(absent | c) / P(absent),  c annotated in this image,

so s = 1 means as likely as base rate and s > 1 means the category fits the
scene. The main quantities are then re-estimated on the plausible and
implausible halves. CPU only; reads existing runs.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


def load_coco(path):
    z = json.load(open(path))
    names = {c["id"]: c["name"] for c in z["categories"]}
    by_img = defaultdict(set)
    for a in z["annotations"]:
        by_img[a["image_id"]].add(names[a["category_id"]])
    return by_img


def cooc(by_img):
    N, M = defaultdict(int), defaultdict(int)
    for s in by_img.values():
        for c in s:
            M[c] += 1
            for d in s:
                N[(c, d)] += 1
    return N, M


def plausibility(absent, ctx, N, M, n_img, alpha=50.0):
    p0 = (M[absent] + 1.0) / (n_img + 1.0)
    vals = [((N[(absent, c)] + alpha * p0) / (M[c] + alpha)) / p0
            for c in ctx if M[c] >= 20]
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    m = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(n_boot)]
    return float(v.mean()), [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def subset_metrics(rows, names, idx):
    lab = lambda p: names[int(np.argmax(p))]
    sub = [rows[i] for i in idx]
    out = {"n": len(sub), "n_scenes": len({r["sample_id"] for r in sub})}
    for key in ("absent|none", "absent|a", "absent|b"):
        ref = [float(lab(r["probs"][key]) == "unknown") for r in sub]
        tgt = [r["absent_hint"] if key != "absent|b" else r["absent_b_hint"]
               for r in sub]
        adopt = [float(lab(r["probs"][key]) == t) for r, t in zip(sub, tgt)]
        out[f"refusal_{key.split('|')[1]}"] = boot_ci(ref)
        out[f"adopt_{key.split('|')[1]}"] = boot_ci(adopt)
    score, label = [], []
    for r in sub:
        score += [r["probs"]["sufficient|none"][names.index("unknown")],
                  r["probs"]["absent|none"][names.index("unknown")]]
        label += [0, 1]
    out["auroc_none"] = float(auroc(score, label))
    return out


def diff_ci(rows, a, b, names, n_boot=2000, seed=1):
    lab = lambda p: names[int(np.argmax(p))]

    def vals(idx, key, tgt):
        return np.array([float(lab(rows[i]["probs"][key])
                              == (rows[i][tgt] if tgt else "unknown"))
                         for i in idx])

    out = {}
    for name, key, tgt in (("refusal_none", "absent|none", None),
                           ("refusal_a", "absent|a", None),
                           ("adopt_a", "absent|a", "absent_hint")):
        va, vb = vals(a, key, tgt), vals(b, key, tgt)
        rng = np.random.default_rng(seed)
        d = [va[rng.integers(0, len(va), len(va))].mean()
             - vb[rng.integers(0, len(vb), len(vb))].mean()
             for _ in range(n_boot)]
        out[name] = (float(va.mean() - vb.mean()),
                     [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))])
    return out


def probe_oof(lens, folds=5, C=0.1):
    z = np.load(lens, allow_pickle=False)
    key = z["key"] if "key" in z.files else z["keys"]
    cond = z["cond"] if "cond" in z.files else np.array(
        ["|".join(k.split("|")[1:]) for k in key])
    H = z["H"].astype(np.float32)
    if H.ndim == 3:
        H = H[:, -1]
    sid = np.array([k.split("|")[0] for k in key])
    keep = np.isin(cond, ["sufficient|none", "absent|none"])
    X = H[keep]
    y = (cond[keep] == "sufficient|none").astype(int)
    g = sid[keep]
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        sc[te] = clf.decision_function(X[te])
    return sc, y, g


def joint_and_cells(rows, names, s_map, s_freq, n_boot=1000, seed=2):
    """Partial effect of scene fit controlling for category frequency."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lab = lambda p: names[int(np.argmax(p))]
    y = np.array([float(lab(r["probs"]["absent|none"]) == "unknown") for r in rows])
    s = np.array([s_map[r["sample_id"]] for r in rows])
    f = np.array([s_freq[r["sample_id"]] for r in rows])
    X = np.column_stack([np.log(np.maximum(s, 1e-4)), np.log(f)])
    X = StandardScaler().fit_transform(X)
    sid = np.array([r["sample_id"] for r in rows])
    u = np.unique(sid)

    def fit(idx):
        clf = LogisticRegression(C=1e4, max_iter=5000).fit(X[idx], y[idx])
        return clf.coef_[0]

    pt = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(u, len(u))
        idx = np.concatenate([np.where(sid == q)[0] for q in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(fit(idx))
    bs = np.array(bs)
    coef = {}
    for j, nm in enumerate(("log_s", "log_freq")):
        coef[nm] = [float(pt[j]), float(np.quantile(bs[:, j], 0.025)),
                    float(np.quantile(bs[:, j], 0.975))]
    rho = float(np.corrcoef(np.argsort(np.argsort(s)),
                            np.argsort(np.argsort(f)))[0, 1])
    sh, fh = s > np.median(s), f > np.median(f)
    cells = {}
    for sn, sm in (("s_lo", ~sh), ("s_hi", sh)):
        for fn, fm in (("f_lo", ~fh), ("f_hi", fh)):
            m = sm & fm
            cells[f"{sn}_{fn}"] = {"n": int(m.sum()),
                                   "refusal_none": float(y[m].mean())}
    return {"coef": coef, "n_boot": int(len(bs)), "spearman_s_freq": rho,
            "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_v5.jsonl")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--train", default="data/coco_val/annotations/instances_train2017.json")
    ap.add_argument("--val", default="data/coco_val/annotations/instances_val2017.json")
    ap.add_argument("--lens", nargs="*", default=[],
                    help="lens npz files aligned with --run, for probe stratification")
    ap.add_argument("--out", default="results/plausibility.json")
    args = ap.parse_args()

    man = [json.loads(l) for l in open(args.manifest)]
    val = load_coco(args.val)
    N, M = cooc(load_coco(args.train))
    n_img = len(val)
    ctx = {img: s for img, s in val.items()}
    s_map = {}
    for r in man:
        s_map[r["sample_id"]] = plausibility(
            r["absent_category"], ctx.get(r["image_id"], set()), N, M, n_img)

    s_freq = {r["sample_id"]: M[r["absent_category"]] / n_img for r in man}

    from .constants import DISTRACTOR
    res = {"n_train_images": n_img, "scores": {}}
    order = sorted(s_map.values())
    q1, q2, q3 = (float(np.quantile(order, q)) for q in (0.25, 0.5, 0.75))
    res["quantiles"] = [q1, q2, q3]

    for path, lens in zip(args.run, args.lens + [None] * len(args.run)):
        lines = [json.loads(l) for l in open(path)]
        meta = lines[0]["meta"]
        names = meta["candidates"].split(",")
        rows = [r for r in lines if "probs" in r]
        rows = [r for r in rows if r["sample_id"] in s_map
                and all(k in r["probs"] for k in
                        ("sufficient|none", "absent|none", "absent|a", "absent|b"))]
        for r in rows:
            r["absent_b_hint"] = DISTRACTOR[r["absent_hint"]]
        s = np.array([s_map[r["sample_id"]] for r in rows])
        idx = np.arange(len(rows))
        halves = {"lo": idx[s <= np.median(s)], "hi": idx[s > np.median(s)]}
        quart = {"q1": idx[s <= q1], "q4": idx[s > q3]}
        f = np.array([s_freq[r["sample_id"]] for r in rows])
        fhalves = {"lo": idx[f <= np.median(f)], "hi": idx[f > np.median(f)]}
        entry = {"model": meta.get("model", path), "n": len(rows),
                 "hi_minus_lo": diff_ci(rows, halves["hi"], halves["lo"], names),
                 "freq_hi_minus_lo": diff_ci(rows, fhalves["hi"], fhalves["lo"], names),
                 "freq_halves": {k: subset_metrics(rows, names, v)
                                 for k, v in fhalves.items()},
                 "n_scenes": len({r["sample_id"] for r in rows}),
                 "halves": {k: subset_metrics(rows, names, v) for k, v in halves.items()},
                 "quartiles": {k: subset_metrics(rows, names, v) for k, v in quart.items()},
                 "joint": joint_and_cells(rows, names, s_map, s_freq)}
        if lens:
            sc, y, g = probe_oof(lens)
            for grp in ("halves", "quartiles"):
                for k, idx in (halves if grp == "halves" else quart).items():
                    ids = {rows[i]["sample_id"] for i in idx}
                    m = np.array([s in ids for s in g])
                    entry[grp][k]["probe_auroc"] = float(auroc(sc[m], y[m]))
                    entry[grp][k]["probe_gap"] = float(
                        sc[m][y[m] == 1].mean() - sc[m][y[m] == 0].mean())
        res["scores"][path] = entry
        print(path)
        for tag in ("hi_minus_lo", "freq_hi_minus_lo"):
            for k, (d, ci) in entry[tag].items():
                print(f"  {tag} {k} {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        for k, v in entry["freq_halves"].items():
            print(f"  freq {k} n={v['n']} ref_none={v['refusal_none'][0]:.3f} "
                  f"ref_a={v['refusal_a'][0]:.3f} adopt_a={v['adopt_a'][0]:.3f}")
        jc = entry["joint"]
        for k, (c, lo, hi) in jc["coef"].items():
            print(f"  joint {k} {c:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  spearman(s,freq) {jc['spearman_s_freq']:+.3f} nboot={jc['n_boot']}")
        for k, v in jc["cells"].items():
            print(f"  cell {k} n={v['n']} ref_none={v['refusal_none']:.3f}")
        for grp in ("halves", "quartiles"):
            for k, v in entry[grp].items():
                print(f"  {grp[:4]} {k} n={v['n']} scenes={v['n_scenes']} "
                      f"ref_none={v['refusal_none'][0]:.3f} "
                      f"ref_a={v['refusal_a'][0]:.3f} ref_b={v['refusal_b'][0]:.3f} "
                      f"adopt_a={v['adopt_a'][0]:.3f} adopt_b={v['adopt_b'][0]:.3f} "
                      f"auroc={v['auroc_none']:.3f}"
                      + (f" probe={v['probe_auroc']:.3f}" if "probe_auroc" in v else ""))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
