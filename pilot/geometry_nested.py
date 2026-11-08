"""Nested bootstrap for the evidence-axis geometry.

The CIs in results/geometry.json resample condition scenes but hold the
whitener W and the no-hint axis w fixed, so they are conditional on those two
estimates. Here every bootstrap resample re-estimates W, d0 and w from the
resampled no-hint pairs before measuring each condition's mean-difference
vector. The resulting intervals include axis-estimation uncertainty.
"""

import argparse
import json

import numpy as np

from .geometry import PAIRS, load, stack, whitener

CONDS = ["none", "correct", "conflict", "conflict_b"]


def metrics(P, A, Xp, Xa, W, w, d0):
    d = Xp.mean(0) - Xa.mean(0)
    nd = float(np.linalg.norm(d)) + 1e-12
    return {
        "ratio": nd / (float(np.linalg.norm(d0)) + 1e-12),
        "angle": float(np.degrees(np.arccos(np.clip(d @ w / nd, -1, 1)))),
        "along_ratio": float((d @ w) / (float(d0 @ w) + 1e-12)),
        "mahal_ratio": float(np.linalg.norm(d @ W) / (float(np.linalg.norm(d0 @ W)) + 1e-12)),
    }


def run(path, n_boot, seed):
    recs, H = load(path)
    P, _ = stack(recs, H, PAIRS["none"][0])
    A, _ = stack(recs, H, PAIRS["none"][1])
    X = np.concatenate([P, A])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(A))])
    W = whitener(X, y)
    d0 = P.mean(0) - A.mean(0)
    w = d0 / (np.linalg.norm(d0) + 1e-12)
    n = len(P)

    Xc = {c: (stack(recs, H, PAIRS[c][0])[0], stack(recs, H, PAIRS[c][1])[0])
          for c in CONDS}
    point = {c: metrics(P, A, *Xc[c], W, w, d0) for c in CONDS}
    dirs = {c: (Xc[c][0].mean(0) - Xc[c][1].mean(0)) for c in CONDS}
    point["cos_a_b"] = float(
        dirs["conflict"] @ dirs["conflict_b"]
        / (np.linalg.norm(dirs["conflict"]) * np.linalg.norm(dirs["conflict_b"])))

    rng = np.random.default_rng(seed)
    draws = {c: {k: [] for k in ("ratio", "angle", "along_ratio", "mahal_ratio")}
             for c in CONDS}
    cos_draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Wb = whitener(X[np.concatenate([idx, idx + n])],
                      y[np.concatenate([idx, idx + n])])
        pb, ab = P[idx], A[idx]
        d0b = pb.mean(0) - ab.mean(0)
        wb = d0b / (np.linalg.norm(d0b) + 1e-12)
        for c in CONDS:
            m = metrics(pb, ab, Xc[c][0][idx], Xc[c][1][idx], Wb, wb, d0b)
            for k, v in m.items():
                draws[c][k].append(v)
        da = Xc["conflict"][0][idx].mean(0) - Xc["conflict"][1][idx].mean(0)
        db = Xc["conflict_b"][0][idx].mean(0) - Xc["conflict_b"][1][idx].mean(0)
        cos_draws.append(float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db))))

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    out = {"n": int(n), "n_boot": int(n_boot), "point": point,
           "nested_ci": {c: {k: ci(v) for k, v in draws[c].items()} for c in CONDS}}
    out["nested_ci"]["cos_a_b"] = ci(cos_draws)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "results/runs/v5_261.jsonl",
        "results/runs/v5_261_7b.jsonl",
        "results/runs/v5_261_llava.jsonl",
        "results/runs/v5_261_gemma.jsonl"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="results/geometry_nested.json")
    args = ap.parse_args()

    res = {}
    for p in args.runs:
        tag = p.split("/")[-1].replace(".jsonl", "")
        res[tag] = run(p, args.n_boot, 0)
        o = res[tag]
        print(tag)
        for c in CONDS:
            print(f"  {c:10s} ratio {o['point'][c]['ratio']:.3f} "
                  f"{o['nested_ci'][c]['ratio']}  angle {o['point'][c]['angle']:.1f} "
                  f"{o['nested_ci'][c]['angle']}  along {o['point'][c]['along_ratio']:.3f} "
                  f"{o['nested_ci'][c]['along_ratio']}", flush=True)
        print(f"  cos(a,b) {o['point']['cos_a_b']:.3f} {o['nested_ci']['cos_a_b']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
