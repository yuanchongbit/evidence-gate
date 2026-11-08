"""Scene-bootstrap intervals for the selective-gate fabrication rates.

The paper reports point estimates at matched answered coverage.  A reviewer
cannot tell whether 0.146 vs 0.169 (7B) is a difference.  This script
resamples scenes, re-runs the matched-coverage selection for every scorer on
the resampled pool, and reports 95% intervals for each fabrication rate and
for the probe-minus-baseline difference.

The out-of-fold probe scores are held fixed across draws (the probe is fitted
once, scene-grouped); the interval therefore covers evaluation sampling, not
probe refitting, which is the same convention as the geometry and AUROC
bootstrap in the paper.
"""

import json

import numpy as np

from .audit import labels, oof_transfer, pairs
from .deploy_check import gate_answered
from .probe import load as load_run

ROOT_PATHS = {
    "3b": "results/runs/v5_261.jsonl",
    "7b": "results/runs/v5_261_7b.jsonl",
    "llava": "results/runs/v5_261_llava.jsonl",
    "gemma": "results/runs/v5_261_gemma.jsonl",
    "phi": "results/runs/phi_261.jsonl",
}
KEYS = ("sufficient|wrong", "absent|a")


def per_scene(recs, names, score):
    rows = []
    for r in recs:
        sc, ab, an = [], [], []
        for k in KEYS:
            sc.append(score[f"{r['sample_id']}|{k}"])
            ab.append(k.startswith("absent"))
            an.append(labels(r, k, names) != "unknown")
        rows.append((r["sample_id"], np.array(sc), np.array(ab, bool),
                     np.array(an, bool)))
    return rows


def select(score, is_absent, answered, coverage):
    return gate_answered(score, is_absent, answered, coverage)


def ci(draws, q=(2.5, 97.5)):
    v = np.array([d for d in draws if d is not None], float)
    return [float(np.quantile(v, q[0] / 100)), float(np.quantile(v, q[1] / 100))]


def main(n_boot=1000, seed=0):
    out = {}
    for tag, path in ROOT_PATHS.items():
        recs, names, H = load_run(str(__import__("pathlib").Path(path)))
        oof = oof_transfer(recs, H, ("sufficient|none", "absent|none"),
                           {"conflict": KEYS})["conflict"]
        p_unk = {f"{r['sample_id']}|{k}": 1.0 - r["probs"][k][-1]
                 for r in recs for k in KEYS}
        mprob = {f"{r['sample_id']}|{k}": max(r["probs"][k][:-1])
                 for r in recs for k in KEYS}
        rows = per_scene(recs, names, oof)
        pu = per_scene(recs, names, p_unk)
        mp = per_scene(recs, names, mprob)
        ids = np.arange(len(rows))
        rng = np.random.default_rng(seed)
        res = {}
        for cov in (0.25, 0.50):
            draws = {k: [] for k in ("probe", "p_unknown", "maxprob", "diff")}
            for _ in range(n_boot):
                pick = rng.choice(ids, len(ids), replace=True)
                sc = np.concatenate([rows[i][1] for i in pick])
                ab = np.concatenate([rows[i][2] for i in pick])
                an = np.concatenate([rows[i][3] for i in pick])
                su = np.concatenate([pu[i][1] for i in pick])
                sm = np.concatenate([mp[i][1] for i in pick])
                a = select(sc, ab, an, cov)
                b = select(su, ab, an, cov)
                c = select(sm, ab, an, cov)
                draws["probe"].append(a["risk_among_answered"])
                draws["p_unknown"].append(b["risk_among_answered"])
                draws["maxprob"].append(c["risk_among_answered"])
                if (a["risk_among_answered"] is not None
                        and b["risk_among_answered"] is not None):
                    draws["diff"].append(a["risk_among_answered"]
                                         - b["risk_among_answered"])
            pt = {}
            full_sc = np.concatenate([r[1] for r in rows])
            full_ab = np.concatenate([r[2] for r in rows])
            full_an = np.concatenate([r[3] for r in rows])
            full_pu = np.concatenate([r[1] for r in pu])
            full_mp = np.concatenate([r[1] for r in mp])
            for k, sc in (("probe", full_sc), ("p_unknown", full_pu),
                          ("maxprob", full_mp)):
                pt[k] = select(sc, full_ab, full_an,
                               cov)["risk_among_answered"]
            res[f"coverage_{cov:.2f}"] = {
                "point": pt,
                "ci": {k: ci(v) for k, v in draws.items()},
                "n_draws": {k: len(v) for k, v in draws.items()},
            }
        out[tag] = res
        print(f"\n{tag}")
        for cov, r in res.items():
            print(f"  {cov}: probe {r['point']['probe']:.3f} "
                  f"{r['ci']['probe']}, p(unknown) {r['point']['p_unknown']:.3f} "
                  f"{r['ci']['p_unknown']}, maxprob {r['point']['maxprob']:.3f} "
                  f"{r['ci']['maxprob']}, diff {r['ci']['diff']}")
    json.dump(out, open("results/gate_ci.json", "w"), indent=1)
    print("\nwrote results/gate_ci.json")


if __name__ == "__main__":
    main()
