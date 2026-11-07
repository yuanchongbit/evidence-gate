"""Offline audit of the v2 claims: refit vs frozen transfer, hint-plausibility
alignment, strict-absent sensitivity, projection matching, deployment shift.

Everything here uses the saved runs and final-layer states; no model forward
pass is needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import DISTRACTOR
from .metrics import auroc
from .probe import load

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "3b": "results/runs/v5_261.jsonl",
    "7b": "results/runs/v5_261_7b.jsonl",
    "llava": "results/runs/v5_261_llava.jsonl",
}
CONFLICT = {"a": ("sufficient|wrong", "absent|a"),
            "b": ("sufficient|wrong", "absent|b")}


def fit(dataset):
    X, y = dataset
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    pipe.fit(X, y)
    return pipe


def pairs(recs, H, pos, neg):
    X, y, g, ids = [], [], [], []
    for r in recs:
        for key, lab in ((pos, 1), (neg, 0)):
            hk = f"{r['sample_id']}|{key}"
            if hk not in H:
                continue
            X.append(H[hk].astype(np.float32))
            y.append(lab)
            g.append(r["sample_id"])
            ids.append((r["sample_id"], key))
    return np.stack(X), np.array(y), np.array(g), ids


def oof_transfer(recs, H, fit_pair, test_pairs):
    """Fit on fit_pair in each scene fold, score all test pairs out of fold."""
    X, y, g, _ = pairs(recs, H, *fit_pair)
    out = {name: {} for name in test_pairs}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        pipe.fit(X[tr], y[tr])
        held = set(g[te])
        for name, (pos, neg) in test_pairs.items():
            for r in recs:
                if r["sample_id"] not in held:
                    continue
                for key, _ in ((pos, 1), (neg, 0)):
                    hk = f"{r['sample_id']}|{key}"
                    if hk in H:
                        out[name][hk] = float(
                            pipe.decision_function(H[hk][None, :])[0])
    return out


def frozen_scores(recs, H, pair, C=0.1):
    """Fit once on the pair and score every state with the frozen pipeline."""
    X, y, _, _ = pairs(recs, H, *pair)
    pipe = fit((X, y))
    return {f"{r['sample_id']}|{k}": float(pipe.decision_function(H[f"{r['sample_id']}|{k}"][None, :])[0])
            for r in recs for k in r["probs"] if f"{r['sample_id']}|{k}" in H}


def boot_mean_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), [float(np.quantile(means, 0.025)),
                                float(np.quantile(means, 0.975))]


def boot_ratio_ci(num, den, n_boot=2000, seed=0):
    """num/den are paired per-scene quantities; bootstrap scenes."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(num), len(num))
        vals.append(num[idx].mean() / den[idx].mean())
    return float(num.mean() / den.mean()), [float(np.quantile(vals, 0.025)),
                                            float(np.quantile(vals, 0.975))]


def boot_auroc_ci(scores, y, g, n_boot=2000, seed=0):
    scores, y, g = np.asarray(scores), np.asarray(y), np.asarray(g)
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        vals.append(auroc(scores[idx], y[idx]))
    return float(auroc(scores, y)), [float(np.quantile(vals, 0.025)),
                                     float(np.quantile(vals, 0.975))]


def probe_metrics(recs, H, subset=None):
    keep = set(subset) if subset is not None else None
    recs = [r for r in recs if keep is None or r["sample_id"] in keep]
    res = {}
    # refit OOF on each condition
    for name, pair in [("none", ("sufficient|none", "absent|none")),
                       ("conflict_a", CONFLICT["a"]),
                       ("conflict_b", CONFLICT["b"]),
                       ("correct", ("sufficient|correct", "absent|none"))]:
        X, y, g, _ = pairs(recs, H, *pair)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.decision_function(X[te])
        a, ci = boot_auroc_ci(oof, y, g)
        res[f"refit_{name}"] = {"auroc": a, "ci": ci}
        # frozen none->name, out of fold
        if name != "none":
            tr_scores = oof_transfer(recs, H, ("sufficient|none", "absent|none"),
                                     {name: pair})[name]
            pos, neg = pair
            ys, ss, gs = [], [], []
            for r in recs:
                for key, lab in ((pos, 1), (neg, 0)):
                    k = f"{r['sample_id']}|{key}"
                    if k in tr_scores:
                        ss.append(tr_scores[k])
                        ys.append(lab)
                        gs.append(r["sample_id"])
            a, ci = boot_auroc_ci(ss, ys, gs)
            res[f"frozen_{name}"] = {"auroc": a, "ci": ci}
        # same-condition frozen (fit on the condition itself)
    # compression gap on the frozen no-hint axis
    none_scores = frozen_scores(recs, H, ("sufficient|none", "absent|none"))
    for name, key in [("a", "absent|a"), ("b", "absent|b")]:
        pres = np.array([none_scores[f"{r['sample_id']}|sufficient|wrong"] for r in recs])
        absent = np.array([none_scores[f"{r['sample_id']}|{key}"] for r in recs])
        base_p = np.array([none_scores[f"{r['sample_id']}|sufficient|none"] for r in recs])
        base_a = np.array([none_scores[f"{r['sample_id']}|absent|none"] for r in recs])
        gap = pres - absent
        base_gap = base_p - base_a
        ratio, ci = boot_ratio_ci(gap, base_gap)
        res[f"gap_ratio_{name}"] = {"ratio": ratio, "ci": ci,
                                    "gap": float(gap.mean()),
                                    "base_gap": float(base_gap.mean())}
    return res


def labels(rec, key, names):
    return names[int(np.argmax(rec["probs"][key]))]


def behaviour(recs, names, subset=None):
    keep = set(subset) if subset is not None else None
    recs = [r for r in recs if keep is None or r["sample_id"] in keep]
    out = {}
    for name, key, hint in [("present_wrong", "sufficient|wrong", DISTRACTOR.__getitem__),
                            ("absent_a", "absent|a", lambda c: None),
                            ("absent_b", "absent|b", lambda c: None)]:
        refuse, adopt, true = [], [], []
        for r in recs:
            lab = labels(r, key, names)
            if key == "sufficient|wrong":
                h = DISTRACTOR[r["gold"]]
            elif key == "absent|a":
                h = r["absent_hint"]
            else:
                h = DISTRACTOR[r["absent_hint"]]
            refuse.append(float(lab == "unknown"))
            adopt.append(float(lab == h))
            true.append(float(lab == r.get("gold")))
        out[name] = {"refusal": float(np.mean(refuse)),
                     "hint": float(np.mean(adopt)),
                     "true": float(np.mean(true))}
    return out


def projection_audit(recs, H):
    from .steer import fit_direction
    w, _ = fit_direction(str(ROOT / "results/runs/v5_261.jsonl"))
    # random orthogonal directions
    rng = np.random.default_rng(0)
    Rs = []
    for _ in range(3):
        v = rng.standard_normal(len(w))
        v -= (v @ w) * w
        for b in Rs:
            v -= (v @ b) * b
        Rs.append((v / np.linalg.norm(v)).astype(np.float32))
    # strong controls: class-mean direction and first PC of the no-hint states
    P = np.stack([H[f"{r['sample_id']}|sufficient|none"] for r in recs] +
                 [H[f"{r['sample_id']}|absent|none"] for r in recs]).astype(np.float32)
    cm = P[:len(recs)].mean(0) - P[len(recs):].mean(0)
    cm = (cm / np.linalg.norm(cm)).astype(np.float32)
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    pc1 = (Vt[0] / np.linalg.norm(Vt[0])).astype(np.float32)
    dirs = [w] + Rs + [cm, pc1]
    keys = [(f"{r['sample_id']}|{k}", k)
            for r in recs for k in ("sufficient|wrong", "absent|a", "absent|b")]
    rows = []
    for hk, k in keys:
        if hk not in H:
            continue
        h = H[hk].astype(np.float32)
        v = np.array([abs(float(h @ d)) for d in dirs])
        rows.append((k, v / np.linalg.norm(h)))
    rows = np.array([[r[1][i] for i in range(len(dirs))] for r in rows])
    names = ["w", "r0", "r1", "r2", "class_mean", "pc1"]
    return {n: {"mean_abs_proj_frac": float(rows[:, i].mean()),
                "median_abs_proj_frac": float(np.median(rows[:, i]))}
            for i, n in enumerate(names)} | {
        "w_over_mean_random": float(rows[:, 0].mean() / rows[:, 1:4].mean())}


def deployment(recs, names, scores):
    """Risk-coverage on the mixed present/absent pool, prevalence shift, and a
    split-conformal risk-control baseline."""
    n = len(recs)
    present = np.array([r["sample_id"] for r in recs])
    absent_ids = present
    # pool: present|wrong items and absent|a items
    item_ids, is_absent = [], []
    for r in recs:
        item_ids.append((r["sample_id"], "sufficient|wrong", False))
        item_ids.append((r["sample_id"], "absent|a", True))
    is_absent = np.array([x[2] for x in item_ids])
    refused = np.array([
        labels(r, k, names) == "unknown" for r in recs
        for k in ("sufficient|wrong", "absent|a")])
    answered = ~refused
    # item order is (present|wrong, absent|a) per scene, i.e. already interleaved
    s = np.array([scores[f"{sid}|{k}"] for sid, k, _ in item_ids])
    p = np.array([1.0 - r["probs"][k][-1] for r in recs
                  for k in ("sufficient|wrong", "absent|a")])
    mp = np.array([max(r["probs"][k]) for r in recs
                   for k in ("sufficient|wrong", "absent|a")])
    mp_col = np.array([max(r["probs"][k][:-1]) for r in recs
                       for k in ("sufficient|wrong", "absent|a")])
    out = {}
    for nm, sc in (("probe", s), ("p_unknown", p), ("maxprob", mp),
                   ("maxprob_colour", mp_col)):
        order = np.argsort(-sc)
        k = len(sc) // 2
        sel = np.zeros(len(sc), bool)
        sel[order[:k]] = True
        sel_abs = sel & is_absent
        sel_pres = sel & ~is_absent
        out[nm] = {
            "coverage": float(sel.mean()),
            "uar": float((sel & is_absent).sum() / max(1, is_absent.sum())),
            "oar": float((sel & ~is_absent & refused).sum()
                         / max(1, (sel & ~is_absent).sum())),
            "fabrication_per_answered": float(
                (sel & is_absent).sum() / max(1, sel.sum())),
            "fabrication_among_model_answered": float(
                (sel & is_absent & answered).sum() / max(1, (sel & answered).sum())),
        }
        # prevalence sweep: keep the balanced 50% threshold, change the mix
        thr = np.quantile(sc, 0.5)
        sweep = {}
        for prev in (0.05, 0.10, 0.25, 0.50):
            # answer probability among absent and present under this threshold
            pa = float(((sc >= thr) & is_absent & answered).sum() / max(1, (is_absent & answered).sum()))
            pp = float(((sc >= thr) & ~is_absent & answered).sum() / max(1, (~is_absent & answered).sum()))
            denom = prev * pa + (1 - prev) * pp
            sweep[str(prev)] = {
                "coverage": float(prev * pa + (1 - prev) * pp),
                "fabrication_among_answered": float(prev * pa / denom) if denom else None}
        out[nm]["prevalence"] = sweep
    # split-conformal risk control: calibrate a threshold on half the scenes so
    # that an upper confidence bound on fabrication among answered absent items
    # is below alpha; evaluate on the other half.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cal, test = perm[:n // 2], perm[n // 2:]
    cal_items = np.concatenate([2 * cal, 2 * cal + 1])
    test_items = np.concatenate([2 * test, 2 * test + 1])
    from scipy.stats import beta
    conformal = {}
    for nm, sc in (("probe", s), ("p_unknown", p), ("maxprob", mp),
                   ("maxprob_colour", mp_col)):
        conformal[nm] = {}
        order = cal_items[np.argsort(-sc[cal_items])]
        # risk upper bound as a function of the answered prefix
        ucb, cov = [], []
        for j in range(1, len(order) + 1):
            sel = order[:j]
            kk = int((is_absent[sel] & answered[sel]).sum())
            nn = len(sel)
            ucb.append(float(beta.ppf(0.95, kk + 1, nn - kk)) if kk < nn else 1.0)
            cov.append(j / len(order))
        ucb = np.asarray(ucb)
        for alpha in (0.05, 0.10):
            feasible = np.where(ucb <= alpha)[0]
            entry = {"alpha": alpha, "infeasible": feasible.size == 0,
                     "calibration_absent": int(is_absent[cal_items].sum()),
                     "min_ucb": float(ucb.min()),
                     "coverage_at_min_ucb": float(cov[int(ucb.argmin())])}
            if feasible.size:
                j = int(feasible[-1]) + 1  # largest answered set meeting the bound
                best = float(sc[order[j - 1]])
                sel = test_items[sc[test_items] >= best]
                entry |= {
                    "threshold": best,
                    "coverage": float(len(sel) / len(test_items)),
                    "uar": float(is_absent[sel].sum() / max(1, is_absent.sum())),
                    "fabrication_per_answered": float(
                        is_absent[sel].sum() / max(1, len(sel))),
                    "false_refusal_of_present": float(
                        (~is_absent[sel] & refused[sel]).sum()
                        / max(1, (~is_absent[sel]).sum()))}
            conformal[nm][str(alpha)] = entry
    out["conformal"] = conformal
    # pollution detector: probe score minus the model's own no-evidence mass.
    # If the hint suppresses both but suppresses the probe more, the gap flags it.
    g = s - p
    out["detector_probe_minus_punknown"] = risk_coverage(g, is_absent)
    return out


def risk_coverage(score, is_absent):
    """50%-coverage fabrication for a monitoring score."""
    k = len(score) // 2
    order = np.argsort(-score)
    sel = np.zeros(len(score), bool)
    sel[order[:k]] = True
    return {
        "coverage": float(sel.mean()),
        "auroc_present_vs_absent": float(auroc(score, (~is_absent).astype(int))),
        "uar": float((sel & is_absent).sum() / max(1, is_absent.sum())),
        "fabrication_per_answered": float((sel & is_absent).sum() / max(1, sel.sum())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/audit_v2.json")
    args = ap.parse_args()
    manifest = [json.loads(l) for l in open(ROOT / "data/manifest_v5.jsonl")]
    presence = {r["sample_id"]: r["p_yes"] for r in
                (json.loads(l) for l in open(ROOT / "results/runs/matched_presence.jsonl"))}
    strict = [m["sample_id"] for m in manifest
              if presence.get(m["sample_id"], 1.0) <= 0.5]
    # The archived presence file covers the category-matched control set, not
    # the main absent categories; strict-absent needs a fresh presence pass.
    strict = strict or None
    result = {"n": len(manifest), "strict_n": None if strict is None else len(strict),
              "models": {}}
    for tag, path in MODELS.items():
        recs, names, H = load(str(ROOT / path))
        res = {"probe": probe_metrics(recs, H)}
        res["behaviour_all"] = behaviour(recs, names)
        res["behaviour_strict"] = None if strict is None else behaviour(recs, names, strict)
        res["probe_strict"] = None if strict is None else probe_metrics(recs, H, strict)
        result["models"][tag] = res
    # projection audit from the 3B states
    recs, names, H = load(str(ROOT / MODELS["3b"]))
    result["projection"] = projection_audit(recs, H)
    # deployment on each model's conflicting pair
    result["deployment"] = {}
    for tag, path in MODELS.items():
        recs, names, H = load(str(ROOT / path))
        scores = frozen_scores(recs, H, ("sufficient|none", "absent|none"))
        result["deployment"][tag] = deployment(recs, names, scores)
    # per-colour labels
    from collections import Counter
    result["gold_counts"] = dict(Counter(r["gold"] for r in recs))
    if args.out:
        with open(ROOT / args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2)[:6000])


if __name__ == "__main__":
    main()
