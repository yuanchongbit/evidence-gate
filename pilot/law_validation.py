"""Out-of-sample checks for the two dilution laws.

law_test.py fits d'_c/d'_0 = a * (gap_c/gap_0) + b on the 12 probe cells and
finds R^2 = 0.918; law_behaviour.py does the same for a behavioural d' on the
8 hinted cells and finds R^2 = 0.57.  Both R^2 are in-sample.  This script
asks whether the relation predicts a cell it was not fitted on:

  * leave-one-cell-out (LOCO): fit slope and intercept on n-1 cells, predict
    the held-out cell;
  * leave-one-model-out (LOMO): fit on the other models, predict the held-out
    model's cells;
  * shared vs per-model gain: one slope for all cells against one slope per
    model (through the origin), scored by RMSE.

The behavioural law carries two cells per model, so a per-model gain is fitted
on two points; LOMO is the test that those constants do not carry over.
"""

import json

import numpy as np

from .law_behaviour import CONDS, MODELS


def load_cells():
    probe = json.load(open("results/law_isotropic.json"))["rows"]
    probe_cells = [(r["model"], r["cond"], r["gap_ratio"], r["dp_ratio"])
                   for r in probe if r["cond"] != "none"]
    beh = json.load(open("results/law_behaviour.json"))["rows"]
    beh_cells = [(tag, cond, beh[tag][cond]["gap_ratio"],
                  beh[tag][cond]["dprime_out_ratio"])
                 for tag in MODELS for cond in CONDS]
    return probe_cells, beh_cells


def fit(x, y, through_origin=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if through_origin:
        a = float((x * y).sum() / (x * x).sum())
        return a, 0.0
    A = np.c_[x, np.ones(len(x))]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def loco(cells):
    errs, base_mean, base_unit = [], [], []
    for i, (tag, cond, x, y) in enumerate(cells):
        tr = [c for j, c in enumerate(cells) if j != i]
        a, b = fit([c[2] for c in tr], [c[3] for c in tr])
        errs.append(y - (a * x + b))
        base_mean.append(y - np.mean([c[3] for c in tr]))
        base_unit.append(y - x)
    errs = np.array(errs)
    return {
        "n": len(cells),
        "mae": float(np.abs(errs).mean()),
        "rmse": float(np.sqrt((errs ** 2).mean())),
        "max_abs": float(np.abs(errs).max()),
        "baseline_training_mean_mae": float(np.abs(base_mean).mean()),
        "baseline_unit_slope_mae": float(np.abs(base_unit).mean()),
    }


def lomo(cells):
    out = {}
    for tag in sorted({c[0] for c in cells}):
        tr = [c for c in cells if c[0] != tag]
        te = [c for c in cells if c[0] == tag]
        a, b = fit([c[2] for c in tr], [c[3] for c in tr])
        errs = [c[3] - (a * c[2] + b) for c in te]
        out[tag] = {"mae": float(np.mean(np.abs(errs))),
                    "errors": [float(e) for e in errs],
                    "slope": a, "intercept": b}
    return out


def shared_vs_per_model(cells):
    tags = sorted({c[0] for c in cells})
    x = np.array([c[2] for c in cells])
    y = np.array([c[3] for c in cells])
    g = np.array([c[0] for c in cells])
    a_shared, _ = fit(x, y, through_origin=True)
    rmse_shared = float(np.sqrt(((y - a_shared * x) ** 2).mean()))
    per = {t: fit([c[2] for c in cells if c[0] == t],
                  [c[3] for c in cells if c[0] == t], through_origin=True)[0]
           for t in tags}
    pred = np.array([per[t] * xx for t, xx in zip(g, x)])
    rmse_per = float(np.sqrt(((y - pred) ** 2).mean()))
    return {"shared_slope": a_shared, "shared_rmse": rmse_shared,
            "per_model_slopes": per, "per_model_rmse": rmse_per}


def bootstrap_probe_slope(n_boot=500, seed=0):
    """Scene bootstrap for the 12-cell probe-law slope, axis re-estimated."""
    from .geometry import PAIRS, load as load_geom, stack

    paths = {
        "3b": "results/runs/v5_261.jsonl",
        "7b": "results/runs/v5_261_7b.jsonl",
        "llava": "results/runs/v5_261_llava.jsonl",
        "gemma": "results/runs/v5_261_gemma.jsonl",
    }
    conds = ["correct", "conflict", "conflict_b"]
    data = {}
    for tag, path in paths.items():
        recs, H = load_geom(path)
        S = {c: (stack(recs, H, PAIRS[c][0])[0], stack(recs, H, PAIRS[c][1])[0])
             for c in ["none"] + conds}
        data[tag] = S

    def law_cells(S, idx):
        P0, A0 = S["none"][0][idx], S["none"][1][idx]
        w = P0.mean(0) - A0.mean(0)
        w = w / np.linalg.norm(w)
        s0 = np.r_[P0 @ w, A0 @ w]
        y0 = np.r_[np.ones(len(P0)), np.zeros(len(A0))]
        sd0 = np.sqrt((((s0[y0 == 1] - s0[y0 == 1].mean()) ** 2).sum()
                       + ((s0[y0 == 0] - s0[y0 == 0].mean()) ** 2).sum())
                      / (len(s0) - 2))
        gap0 = (P0 @ w).mean() - (A0 @ w).mean()
        dp0 = gap0 / sd0
        out = []
        for c in conds:
            P, A = S[c][0][idx], S[c][1][idx]
            s = np.r_[P @ w, A @ w]
            y = np.r_[np.ones(len(P)), np.zeros(len(A))]
            sd = np.sqrt((((s[y == 1] - s[y == 1].mean()) ** 2).sum()
                          + ((s[y == 0] - s[y == 0].mean()) ** 2).sum())
                         / (len(s) - 2))
            gap = (P @ w).mean() - (A @ w).mean()
            out.append((gap / gap0, (gap / sd) / dp0))
        return out

    rng = np.random.default_rng(seed)
    slopes = []
    n = len(data["3b"]["none"][0])
    for _ in range(n_boot):
        bx, by = [], []
        for tag, S in data.items():
            idx = rng.integers(0, n, n)
            for x, y in law_cells(S, idx):
                bx.append(x)
                by.append(y)
        slopes.append(fit(bx, by, through_origin=True)[0])
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return {"n_boot": n_boot, "mean": float(np.mean(slopes)),
            "ci": [float(lo), float(hi)]}


def main():
    probe_cells, beh_cells = load_cells()
    out = {}
    for name, cells in [("probe", probe_cells), ("behaviour", beh_cells)]:
        loco_fit = loco(cells)
        lomo_fit = lomo(cells)
        svp = shared_vs_per_model(cells)
        out[name] = {"cells": cells, "loco": loco_fit, "lomo": lomo_fit,
                     "shared_vs_per_model": svp}
        print(f"\n== {name} (n={len(cells)}) ==")
        print(f"LOCO: MAE {loco_fit['mae']:.3f}, RMSE {loco_fit['rmse']:.3f}, "
              f"max {loco_fit['max_abs']:.3f}")
        print(f"  baselines: training mean {loco_fit['baseline_training_mean_mae']:.3f}, "
              f"unit slope {loco_fit['baseline_unit_slope_mae']:.3f}")
        for tag, v in lomo_fit.items():
            print(f"  LOMO {tag:6s}: MAE {v['mae']:.3f}  errors {v['errors']}")
        print(f"  shared slope {svp['shared_slope']:.3f} RMSE {svp['shared_rmse']:.3f} | "
              f"per-model {svp['per_model_slopes']} RMSE {svp['per_model_rmse']:.3f}")
    boot = bootstrap_probe_slope()
    out["probe_bootstrap_slope"] = boot
    print(f"\nprobe slope, scene bootstrap ({boot['n_boot']} draws, axis re-estimated): "
          f"{boot['mean']:.3f}, 95% CI {boot['ci']}")
    json.dump(out, open("results/law_validation.json", "w"), indent=1)
    print("\nwrote results/law_validation.json")


if __name__ == "__main__":
    main()
