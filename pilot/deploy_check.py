"""Deployment re-audit: scene-cross-fitted gate scores and valid risk control.

Two protocol fixes relative to pilot/audit.py:

1. The 50%-coverage gate is scored out of fold: the probe is fitted on the
   no-hint pairs of four scene folds and applied to the conflict pairs of the
   held-out fold, so no evaluation scene contributes a label to the fit.

2. The conformal risk bound uses the answered subset in both numerator and
   denominator (risk among answered items) and a Clopper-Pearson upper bound
   corrected over the scanned threshold grid (Bonferroni). The old audit
   counted refused absent items in the denominator but not the numerator,
   which understates risk.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .audit import MODELS, labels, oof_transfer, pairs
from .probe import load

ROOT = Path(__file__).resolve().parent.parent
MORE = {
    "gemma": "results/runs/v5_261_gemma.jsonl",
    "phi": "results/runs/phi_261.jsonl",
}


def item_arrays(recs, names, score):
    sc, is_absent, answered = [], [], []
    for r in recs:
        for k in ("sufficient|wrong", "absent|a"):
            sc.append(score[f"{r['sample_id']}|{k}"])
            is_absent.append(k.startswith("absent"))
            answered.append(labels(r, k, names) != "unknown")
    return np.array(sc), np.array(is_absent, bool), np.array(answered, bool)


def risk_at(sel, is_absent, answered):
    ans = answered[sel]
    n = int(ans.sum())
    if n == 0:
        return None, 0
    return float((is_absent[sel] & ans).sum() / n), n


def gate(score, is_absent, answered, coverage=0.5):
    order = np.argsort(-score)
    k = int(coverage * len(score))
    sel = np.zeros(len(score), bool)
    sel[order[:k]] = True
    risk, n = risk_at(sel, is_absent, answered)
    return {"coverage": float(k / len(score)), "risk_among_answered": risk,
            "n_answered": n,
            "fabrication_per_selected": float(
                (sel & is_absent).sum() / max(1, sel.sum()))}


def gate_answered(score, is_absent, answered, coverage=0.5):
    """Select a prefix by score until `coverage` of all items are answered.

    Selection coverage and answered coverage coincide only when the model
    answers every selected item.  Here the model may still emit `unknown` on a
    selected item, so we choose the prefix whose cumulative answered count
    reaches the target.  This puts every scorer on the same achieved coverage.
    """
    order = np.argsort(-score, kind="stable")
    target = int(round(coverage * len(score)))
    cum = np.cumsum(answered[order])
    k = int(np.searchsorted(cum, target, side="left") + 1)
    k = min(k, len(score))
    sel = np.zeros(len(score), bool)
    sel[order[:k]] = True
    risk, n = risk_at(sel, is_absent, answered)
    return {"answered_coverage": float(n / len(score)),
            "selection_coverage": float(k / len(score)),
            "risk_among_answered": risk, "n_answered": n,
            "fabrication_per_selected": float(
                (sel & is_absent).sum() / max(1, sel.sum()))}


def prevalence_sweep(score, is_absent, answered, threshold, prevs):
    """Rewrite the absent:present mix while holding the gate threshold fixed."""
    out = {}
    ans_abs = answered & is_absent
    ans_pre = answered & ~is_absent
    p_ans_abs = float((ans_abs & (score >= threshold)).sum() / max(1, ans_abs.sum()))
    p_ans_pre = float((ans_pre & (score >= threshold)).sum() / max(1, ans_pre.sum()))
    for prev in prevs:
        denom = prev * p_ans_abs + (1 - prev) * p_ans_pre
        out[f"{prev:.2f}"] = {
            "coverage": denom,
            "fabrication_among_answered": float(prev * p_ans_abs / denom)
            if denom else None}
    return out


def conformal(cal_scores, test_scores, cal_absent, cal_ans,
              test_absent, test_ans, alpha, n_grid=50):
    # thresholds at equally spaced quantiles of the calibration scores
    qs = np.linspace(0, 1, n_grid)
    thrs = np.unique(np.quantile(cal_scores, qs))
    delta = 0.05 / len(thrs)  # family-wise over the scanned grid
    rows = []
    for t in thrs:
        sel = cal_scores >= t
        ans = cal_ans[sel]
        n = int(ans.sum())
        if n == 0:
            continue
        kk = int((cal_absent[sel] & ans).sum())
        risk = kk / n
        rows.append({
            "threshold": float(t),
            "coverage": float(sel.mean()),
            "risk": risk,
            "n_answered": n,
            "ucb_consistent": float(beta.ppf(0.95, kk + 1, n - kk))
            if kk < n else 1.0,
            "ucb_cp_fw": float(beta.ppf(1.0 - delta, kk + 1, n - kk))
            if kk < n else 1.0,
            "ucb_cp_point": float(beta.ppf(0.95, kk + 1, n - kk))
            if kk < n else 1.0,
        })
    point = [r for r in rows if r["ucb_cp_point"] <= alpha]
    point_info = None
    if point:
        bp = max(point, key=lambda r: r["coverage"])
        sel = test_scores >= bp["threshold"]
        rp, np_ = risk_at(sel, test_absent, test_ans)
        point_info = {"calibration_coverage": bp["coverage"],
                      "test_coverage": float(sel.mean()),
                      "test_risk_among_answered": rp}
    feasible = [r for r in rows if r["ucb_cp_fw"] <= alpha]
    if not feasible:
        return {"alpha": alpha, "feasible": False,
                "min_ucb": float(min(r["ucb_cp_fw"] for r in rows)),
                "pointwise": point_info,
                "grid": n_grid}
    best = max(feasible, key=lambda r: r["coverage"])
    sel = test_scores >= best["threshold"]
    risk, n = risk_at(sel, test_absent, test_ans)
    return {"alpha": alpha, "feasible": True,
            "threshold": best["threshold"],
            "calibration_coverage": best["coverage"],
            "calibration_ucb": best["ucb_cp_fw"],
            "pointwise": point_info,
            "test_coverage": float(sel.mean()),
            "test_risk_among_answered": risk,
            "test_n_answered": n,
            "test_fabrication_per_selected": float(
                (sel & test_absent).sum() / max(1, sel.sum())),
            "grid": n_grid}


def run_model(tag, path, seeds=20):
    recs, names, H = load(str(ROOT / path))
    # cross-fitted scores: no-hint fit, conflict score, held-out scenes
    oof = oof_transfer(recs, H, ("sufficient|none", "absent|none"),
                       {"conflict": ("sufficient|wrong", "absent|a")})["conflict"]
    X, y, g, _ = pairs(recs, H, "sufficient|none", "absent|none")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    insample = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    insample.fit(X, y)
    ins = {f"{r['sample_id']}|{k}": float(insample.decision_function(
        H[f"{r['sample_id']}|{k}"][None, :])[0])
        for r in recs for k in ("sufficient|wrong", "absent|a")}
    sc_o, ab_o, an_o = item_arrays(recs, names, oof)
    sc_i, ab_i, an_i = item_arrays(recs, names, ins)

    out = {"gate_oof": gate(sc_o, ab_o, an_o),
           "gate_insample": gate(sc_i, ab_i, an_i),
           "gate_oof_70": gate(sc_o, ab_o, an_o, 0.7),
           "gate_insample_70": gate(sc_i, ab_i, an_i, 0.7),
           "gate_oof_matched50": gate_answered(sc_o, ab_o, an_o),
           "gate_insample_matched50": gate_answered(sc_i, ab_i, an_i),
           "gate_oof_matched25": gate_answered(sc_o, ab_o, an_o, 0.25),
           "gate_insample_matched25": gate_answered(sc_i, ab_i, an_i, 0.25)}
    # baselines on the same pool and convention
    p_unk = np.array([1.0 - r["probs"][k][-1] for r in recs
                      for k in ("sufficient|wrong", "absent|a")])
    mprob = np.array([max(r["probs"][k]) for r in recs
                      for k in ("sufficient|wrong", "absent|a")])
    mprob_col = np.array([max(r["probs"][k][:-1]) for r in recs
                          for k in ("sufficient|wrong", "absent|a")])
    for nm, sc in (("p_unknown", p_unk), ("maxprob", mprob),
                   ("maxprob_colour", mprob_col)):
        out[f"gate_{nm}"] = gate(sc, ab_o, an_o)
        out[f"gate_{nm}_70"] = gate(sc, ab_o, an_o, 0.7)
        out[f"gate_{nm}_matched50"] = gate_answered(sc, ab_o, an_o)
        out[f"gate_{nm}_matched25"] = gate_answered(sc, ab_o, an_o, 0.25)

    # Prevalence sweep at a threshold chosen for 25% answered coverage on the
    # balanced pool (feasible for every model; 7B cannot reach 50%).
    prevs = (0.05, 0.10, 0.25, 0.50)
    for nm, sc in (("oof", sc_o), ("p_unknown", p_unk),
                   ("maxprob_colour", mprob_col)):
        m = gate_answered(sc, ab_o, an_o, 0.25)
        order = np.argsort(-sc, kind="stable")
        cum = np.cumsum(an_o[order])
        k = min(int(np.searchsorted(cum, int(round(0.25 * len(sc)))) + 1), len(sc))
        thr = float(sc[order[k - 1]])
        out[f"prevalence_{nm}"] = {
            "threshold": thr,
            "matched25": m,
            "sweep": prevalence_sweep(sc, ab_o, an_o, thr, prevs)}

    # split conformal on cross-fitted scores: each scene is scored by a probe
    # that never saw it, so calibration and test scores are exchangeable.
    ids = np.array(sorted(set(r["sample_id"] for r in recs)))
    sid_items = np.repeat(np.array([r["sample_id"] for r in recs]), 2)
    conf = {}
    for alpha in (0.05, 0.10):
        vals = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(ids))
            cal = set(ids[perm[: len(ids) // 2]])
            test = set(ids[perm[len(ids) // 2:]])
            cal_mask = np.isin(sid_items, list(cal))
            test_mask = np.isin(sid_items, list(test))
            vals.append(conformal(sc_o[cal_mask], sc_o[test_mask],
                                  ab_o[cal_mask], an_o[cal_mask],
                                  ab_o[test_mask], an_o[test_mask], alpha))
        feas = [v for v in vals if v["feasible"]]
        out[f"conformal_{int(alpha*100)}"] = {
            "n_splits": len(vals), "n_feasible": len(feas),
            "test_risk_mean": float(np.mean([v["test_risk_among_answered"]
                                             for v in feas])) if feas else None,
            "test_coverage_mean": float(np.mean([v["test_coverage"]
                                                 for v in feas])) if feas else None,
            "min_ucb_mean": float(np.mean([v["min_ucb"] for v in vals
                                           if "min_ucb" in v])) if not feas else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deploy_check.json")
    ap.add_argument("--models", default="3b,7b,llava",
                    help="comma-separated tags from MODELS/MORE")
    args = ap.parse_args()
    paths = dict(MODELS)
    paths.update(MORE)
    tags = [t for t in args.models.split(",") if t]
    res = {tag: run_model(tag, paths[tag]) for tag in tags}
    with open(ROOT / args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
