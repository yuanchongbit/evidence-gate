"""Does the geometric dilution law predict behaviour, not just probe d'?

The paper's law regresses the fixed-probe d' ratio on the along-axis mean-gap
ratio: d'/d'0 = 1.42 * gap ratio over 12 model-condition cells.  Because
d' = gap / spread is an identity, its empirical content is the near-constancy
of the within-class spread contraction, and the dependent variable is still
the probe.  This script replaces the dependent variable with output-space
quantities that use no hidden states at all:

  * d'_out: from the logit of p(unknown), d' = (m_abs - m_pre) / pooled sd.
    Continuous, so it does not saturate like refusal rates do.
  * d'_rate: from argmax refusal rates, z(hit) - z(false alarm), Haldane
    corrected.  This is the behavioural d' the paper already reports.
  * AUROC of p(unknown), converted to d' = sqrt(2) * Phi^-1(AUROC).

Only the two conflicting hints enter: they are the cells in which the absent
arm is actually hinted, so the present/absent contrast is within-hint and
the same pairs define the geometric ratio.  The correct-hint cell has no
hinted absent arm and would mix hint and no-hint in the behavioural contrast.

The predictor is the along-axis mean-gap ratio on the same pair: projection
of the condition's mean difference on the no-hint axis, over the no-hint
projection.  The scene bootstrap re-estimates the axis inside every draw.
"""

import json

import numpy as np
from scipy.stats import norm

from .geometry import PAIRS, load, stack

MODELS = {
    "3b": "results/runs/v5_261.jsonl",
    "7b": "results/runs/v5_261_7b.jsonl",
    "llava": "results/runs/v5_261_llava.jsonl",
    "gemma": "results/runs/v5_261_gemma.jsonl",
}
CONDS = ["conflict", "conflict_b"]


def p_unknown(probs):
    return probs[-1]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def behavioural(recs, present_key, absent_key):
    pre = np.array([logit(p_unknown(r["probs"][present_key]))
                    for r in recs if present_key in r["probs"]])
    abst = np.array([logit(p_unknown(r["probs"][absent_key]))
                     for r in recs if absent_key in r["probs"]])
    sd = np.sqrt((pre.var(ddof=1) + abst.var(ddof=1)) / 2)
    d_out = float((abst.mean() - pre.mean()) / sd)
    y = np.r_[np.ones(len(pre)), np.zeros(len(abst))]
    s = np.r_[abst, pre]
    auroc = float((s[y == 1][:, None] > s[y == 0][None, :]).mean()
                  + 0.5 * (s[y == 1][:, None] == s[y == 0][None, :]).mean())
    fa = float(np.mean([np.argmax(r["probs"][present_key]) == len(r["probs"][present_key]) - 1
                        for r in recs if present_key in r["probs"]]))
    hit = float(np.mean([np.argmax(r["probs"][absent_key]) == len(r["probs"][absent_key]) - 1
                         for r in recs if absent_key in r["probs"]]))
    n1, n2 = len(pre), len(abst)
    fa_c = (fa * n1 + 0.5) / (n1 + 1.0)
    hit_c = (hit * n2 + 0.5) / (n2 + 1.0)
    return {
        "dprime_out": d_out,
        "dprime_rate": float(norm.ppf(hit_c) - norm.ppf(fa_c)),
        "auroc": auroc,
        "auroc_dprime": float(np.sqrt(2) * norm.ppf(auroc)),
        "refuse_present": fa,
        "refuse_absent": hit,
        "mean_logit_present": float(pre.mean()),
        "mean_logit_absent": float(abst.mean()),
        "sd_logit": float(sd),
    }


def axis(recs, H):
    Xn, _ = stack(recs, H, PAIRS["none"][0])
    An, _ = stack(recs, H, PAIRS["none"][1])
    d = Xn.mean(0) - An.mean(0)
    return d / np.linalg.norm(d)


def gap_ratio(recs, H, cond, w):
    Xn, _ = stack(recs, H, PAIRS["none"][0])
    An, _ = stack(recs, H, PAIRS["none"][1])
    Xc, _ = stack(recs, H, PAIRS[cond][0])
    Ac, _ = stack(recs, H, PAIRS[cond][1])
    g0 = float((Xn @ w).mean() - (An @ w).mean())
    return float((Xc @ w).mean() - (Ac @ w).mean()) / g0


def cells(recs, H):
    w = axis(recs, H)
    base = behavioural(recs, PAIRS["none"][0], PAIRS["none"][1])
    out = {}
    for cond in CONDS:
        b = behavioural(recs, *PAIRS[cond])
        out[cond] = {
            **b,
            "dprime_out_ratio": b["dprime_out"] / base["dprime_out"],
            "dprime_rate_ratio": b["dprime_rate"] / base["dprime_rate"],
            "auroc_dprime_ratio": b["auroc_dprime"] / base["auroc_dprime"],
            "gap_ratio": gap_ratio(recs, H, cond, w),
        }
    out["none"] = {**base, "dprime_out_ratio": 1.0,
                   "dprime_rate_ratio": 1.0, "auroc_dprime_ratio": 1.0,
                   "gap_ratio": 1.0}
    return out


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.c_[x, np.ones(len(x))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
    s0 = float((x * y).sum() / (x * x).sum())
    return {"slope": float(coef[0]), "intercept": float(coef[1]),
            "r2": float(r2), "through_origin_slope": s0,
            "through_origin_rmse": float(np.sqrt(((y - s0 * x) ** 2).mean())),
            "n": len(x)}


def ratio_slope(rows, key):
    x = [rows[t][c]["gap_ratio"] for t in MODELS for c in CONDS]
    y = [rows[t][c][key] for t in MODELS for c in CONDS]
    return slope(x, y)


def main():
    rows = {}
    data = {}
    for tag, path in MODELS.items():
        recs, H = load(path)
        data[tag] = (recs, H)
        rows[tag] = cells(recs, H)
        for cond in CONDS:
            c = rows[tag][cond]
            print(f"{tag:6s} {cond:11s} gap={c['gap_ratio']:.3f} "
                  f"d'out={c['dprime_out']:.2f} ratio={c['dprime_out_ratio']:.3f} "
                  f"| d'rate={c['dprime_rate']:.2f} "
                  f"ratio={c['dprime_rate_ratio']:.3f} "
                  f"| AUROC={c['auroc']:.3f}")

    fits = {}
    for key in ["dprime_out_ratio", "dprime_rate_ratio", "auroc_dprime_ratio"]:
        fit = ratio_slope(rows, key)
        fits[key] = fit
        print(f"\n{key}: d' ratio = {fit['slope']:.3f} gap + {fit['intercept']:.3f}, "
              f"R2={fit['r2']:.3f}; through origin {fit['through_origin_slope']:.3f} "
              f"(RMSE {fit['through_origin_rmse']:.3f}), n={fit['n']}")

    per_model = {}
    for tag in MODELS:
        gains = {k: [rows[tag][c][k] / rows[tag][c]["gap_ratio"] for c in CONDS]
                 for k in ["dprime_out_ratio", "dprime_rate_ratio"]}
        per_model[tag] = {k: {"mean": float(np.mean(v)), "min": float(min(v)),
                              "max": float(max(v))} for k, v in gains.items()}
        print(f"  {tag:6s} d'out/gap={np.mean(gains['dprime_out_ratio']):.3f} "
              f"d'rate/gap={np.mean(gains['dprime_rate_ratio']):.3f}")

    # scene bootstrap, pooled law on d'_out
    rng = np.random.default_rng(0)
    ids = {tag: np.array([r["sample_id"] for r in recs])
           for tag, (recs, _) in data.items()}
    by_id = {tag: {i: [r for r in recs if r["sample_id"] == i] for i in ids[tag]}
             for tag, (recs, _) in data.items()}
    boot = {"dprime_out_ratio": [], "dprime_rate_ratio": []}
    for _ in range(500):
        bx, by_out, by_rate = [], [], []
        for tag, (recs, H) in data.items():
            pick = rng.choice(ids[tag], len(ids[tag]), replace=True)
            sample = [r for i in pick for r in by_id[tag][i]]
            cell = cells(sample, H)
            for cond in CONDS:
                bx.append(cell[cond]["gap_ratio"])
                by_out.append(cell[cond]["dprime_out_ratio"])
                by_rate.append(cell[cond]["dprime_rate_ratio"])
        boot["dprime_out_ratio"].append(slope(bx, by_out)["slope"])
        boot["dprime_rate_ratio"].append(slope(bx, by_rate)["slope"])
    for key, vals in boot.items():
        lo, hi = np.quantile(vals, [0.025, 0.975])
        fits[key]["bootstrap_slope"] = float(np.mean(vals))
        fits[key]["bootstrap_slope_ci"] = [float(lo), float(hi)]
        print(f"{key}: bootstrap slope {np.mean(vals):.3f} [{lo:.3f}, {hi:.3f}]")

    out = {"rows": rows, "fit": fits, "per_model": per_model}
    json.dump(out, open("results/law_behaviour.json", "w"), indent=1)
    print("\nwrote results/law_behaviour.json")


if __name__ == "__main__":
    main()
