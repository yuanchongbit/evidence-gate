"""Second-stimulus-set analysis: behaviour, probe, geometry, SDT and gate.

Self-contained so it can run against a run directory that only has
pilot.geometry / pilot.metrics / pilot.constants available.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from pilot.constants import CANDIDATES, DISTRACTOR
from pilot.geometry import PAIRS, load, stack
from pilot.metrics import auroc

CONFLICT = ("sufficient|wrong", "absent|a")


def refuses(p):
    return int(np.argmax(p) == len(p) - 1)


def answer(p):
    return CANDIDATES[int(np.argmax(p))]


def behaviour(recs):
    out = {}
    for r in recs:
        for k, p in r["probs"].items():
            d = out.setdefault(k, {"n": 0, "refuse": 0, "gold": 0, "adopt": 0})
            d["n"] += 1
            d["refuse"] += refuses(p)
            a = answer(p)
            d["gold"] += int(a == r["gold"])
            if k == "sufficient|correct":
                d["adopt"] += int(a == r["gold"])
            elif k == "sufficient|wrong":
                d["adopt"] += int(a == r["distractor"])
            elif k == "absent|a":
                d["adopt"] += int(a == r["absent_hint"])
            elif k == "absent|b":
                d["adopt"] += int(a == DISTRACTOR[r["absent_hint"]])
    return {k: {kk: (v / d["n"] if kk != "n" else v) for kk, v in d.items()}
            for k, d in out.items()}


def gkfold():
    return GroupKFold(n_splits=5)


def probe_dataset(recs, H, pos, neg):
    X, y, g = [], [], []
    for r in recs:
        for key, lab in ((pos, 1), (neg, 0)):
            hk = f"{r['sample_id']}|{key}"
            if hk in H:
                X.append(H[hk].astype(np.float32))
                y.append(lab)
                g.append(r["sample_id"])
    return np.stack(X), np.array(y), np.array(g)


def oof_scores(X, y, g, C=0.1):
    oof = np.zeros(len(y))
    for tr, te in gkfold().split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def scene_boot_auroc(scores, y, g, n_boot=1000, seed=0):
    uniq = np.unique(g)
    idx_of = {k: np.where(g == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[k] for k in pick])
        vals.append(auroc(scores[idx], y[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def probe_block(recs, H):
    Xn, yn, gn = probe_dataset(recs, H, *PAIRS["none"])
    oof_n = oof_scores(Xn, yn, gn)
    Xh, yh, gh = probe_dataset(recs, H, *PAIRS["conflict"])
    oof_h = oof_scores(Xh, yh, gh)
    out = {"none_auroc": float(auroc(oof_n, yn)),
           "none_ci": scene_boot_auroc(oof_n, yn, gn),
           "hint_auroc": float(auroc(oof_h, yh)),
           "hint_ci": scene_boot_auroc(oof_h, yh, gh),
           "conflict_auroc": float(auroc(oof_h, yh)),
           "conflict_ci": scene_boot_auroc(oof_h, yh, gh),
           "n_none": int(len(yn)), "n_hint": int(len(yh))}
    return out


def pooled_sd(s, y):
    return float(np.sqrt((((s[y == 1] - s[y == 1].mean()) ** 2).sum()
                          + ((s[y == 0] - s[y == 0].mean()) ** 2).sum())
                         / (len(s) - 2)))


def law_cells(recs, H, idx=None):
    Xn, _ = stack(recs, H, PAIRS["none"][0])
    An, _ = stack(recs, H, PAIRS["none"][1])
    if idx is not None:
        Xn, An = Xn[idx], An[idx]
    d0 = Xn.mean(0) - An.mean(0)
    w = d0 / (np.linalg.norm(d0) + 1e-12)
    s0 = np.r_[Xn @ w, An @ w]
    y0 = np.r_[np.ones(len(Xn)), np.zeros(len(An))]
    gap0 = float((Xn @ w).mean() - (An @ w).mean())
    sd0 = pooled_sd(s0, y0)
    out = {}
    for name, (pk, nk) in PAIRS.items():
        if name == "none":
            continue
        Xp, _ = stack(recs, H, pk)
        Xa, _ = stack(recs, H, nk)
        if idx is not None:
            Xp, Xa = Xp[idx], Xa[idx]
        s = np.r_[Xp @ w, Xa @ w]
        y = np.r_[np.ones(len(Xp)), np.zeros(len(Xa))]
        gap = float((Xp @ w).mean() - (Xa @ w).mean())
        sd = pooled_sd(s, y)
        out[name] = {"gap_ratio": gap / gap0, "sd_ratio": sd / sd0,
                     "dp_ratio": (gap / sd) / (gap0 / sd0),
                     "auc": float(auroc(s, y))}
    return out


def law_block(recs, H, n_boot=500):
    point = law_cells(recs, H)
    rng = np.random.default_rng(0)
    n = len(recs)
    draws = {c: {k: [] for k in point[c]} for c in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for c, v in law_cells(recs, H, idx).items():
            for k, x in v.items():
                draws[c][k].append(x)
    out = {}
    for c, v in point.items():
        ci = {k: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
              for k, x in draws[c].items()}
        out[c] = {"point": v, "ci": ci,
                  "predicted_dp_ratio": 1.425 * v["gap_ratio"],
                  "residual": v["dp_ratio"] - 1.425 * v["gap_ratio"]}
    return out


def binary_sdt(recs, conditions):
    out = {}
    for c, (pos_key, neg_keys) in conditions.items():
        pres = [r["probs"][pos_key] for r in recs if pos_key in r["probs"]]
        neg = [r["probs"][k] for r in recs for k in neg_keys if k in r["probs"]]
        fa = np.mean([refuses(p) for p in pres])
        hit = np.mean([refuses(p) for p in neg])
        n1, n2 = len(pres), len(neg)
        fa_c = (fa * n1 + 0.5) / (n1 + 1.0)
        hit_c = (hit * n2 + 0.5) / (n2 + 1.0)
        out[c] = {"n_present": n1, "n_absent": n2,
                  "fa_refuse": float(fa), "hit_refuse": float(hit),
                  "dprime": float(norm.ppf(hit_c) - norm.ppf(fa_c)),
                  "criterion": float(-0.5 * (norm.ppf(hit_c) + norm.ppf(fa_c)))}
    return out


COND_SDT = {
    "none": ("sufficient|none", ["absent|none"]),
    "conflict": ("sufficient|wrong", ["absent|a", "absent|b"]),
    "correct": ("sufficient|correct", ["absent|none"]),
}


def sdt_block(recs, n_boot=1000):
    point = binary_sdt(recs, COND_SDT)
    ids = np.array([r["sample_id"] for r in recs])
    uniq = np.unique(ids)
    by_id = {i: [r for r in recs if r["sample_id"] == i] for i in uniq}
    rng = np.random.default_rng(0)
    draws = {c: {"dprime": [], "criterion": [], "dprime_diff": [],
                 "criterion_diff": []} for c in COND_SDT}
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sample = [r for i in pick for r in by_id[i]]
        s = binary_sdt(sample, COND_SDT)
        for c in COND_SDT:
            draws[c]["dprime"].append(s[c]["dprime"])
            draws[c]["criterion"].append(s[c]["criterion"])
            draws[c]["dprime_diff"].append(s[c]["dprime"] - s["none"]["dprime"])
            draws[c]["criterion_diff"].append(
                s[c]["criterion"] - s["none"]["criterion"])
    ci = {c: {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
              for k, v in d.items()} for c, d in draws.items()}
    return {"point": point, "ci": ci}


def gate_answered(score, is_absent, answered, coverage):
    order = np.argsort(-score, kind="stable")
    target = int(round(coverage * len(score)))
    cum = np.cumsum(answered[order])
    k = int(np.searchsorted(cum, target, side="left") + 1)
    k = min(k, len(score))
    sel = np.zeros(len(score), bool)
    sel[order[:k]] = True
    ans = answered[sel]
    n = int(ans.sum())
    risk = float((is_absent[sel] & ans).sum() / n) if n else None
    return {"answered_coverage": float(n / len(score)),
            "risk_among_answered": risk, "n_answered": n}


def gate_block(recs, H, n_boot=1000):
    Xn, yn, gn = probe_dataset(recs, H, *PAIRS["none"])
    oof_n = oof_scores(Xn, yn, gn)
    # scene-held-out probe scores on the conflict items
    scores_probe = {}
    for tr, te in gkfold().split(Xn, yn, gn):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(Xn[tr], yn[tr])
        held = set(gn[te])
        for r in recs:
            if r["sample_id"] not in held:
                continue
            for key in CONFLICT:
                hk = f"{r['sample_id']}|{key}"
                if hk in H:
                    scores_probe[hk] = float(
                        clf.decision_function(H[hk][None, :])[0])
    sp = np.array([scores_probe[f"{r['sample_id']}|{k}"] for r in recs for k in CONFLICT])
    su = np.array([1.0 - r["probs"][k][-1] for r in recs for k in CONFLICT])
    sm = np.array([max(r["probs"][k][:-1]) for r in recs for k in CONFLICT])
    ab = np.array([k.startswith("absent") for r in recs for k in CONFLICT], bool)
    an = np.array([not refuses(r["probs"][k]) for r in recs for k in CONFLICT], bool)
    scene = np.array([r["sample_id"] for r in recs for k in CONFLICT])
    uniq = np.unique(scene)
    idx_of = {k: np.where(scene == k)[0] for k in uniq}
    rng = np.random.default_rng(0)
    out = {}
    for cov in (0.25, 0.50):
        pt = {name: gate_answered(s, ab, an, cov)
              for name, s in (("probe", sp), ("p_unknown", su), ("maxprob", sm))}
        draws = {k: [] for k in ("probe", "p_unknown", "maxprob", "diff")}
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([idx_of[k] for k in pick])
            a = gate_answered(sp[idx], ab[idx], an[idx], cov)
            b = gate_answered(su[idx], ab[idx], an[idx], cov)
            c = gate_answered(sm[idx], ab[idx], an[idx], cov)
            draws["probe"].append(a["risk_among_answered"])
            draws["p_unknown"].append(b["risk_among_answered"])
            draws["maxprob"].append(c["risk_among_answered"])
            if a["risk_among_answered"] is not None and b["risk_among_answered"] is not None:
                draws["diff"].append(a["risk_among_answered"] - b["risk_among_answered"])
        ci = {}
        for k, v in draws.items():
            v = np.array([x for x in v if x is not None], float)
            ci[k] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
        out[f"coverage_{cov:.2f}"] = {"point": pt, "ci": ci}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="/tmp/voc_results.json")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    recs, H = load(args.run)
    print(f"n={len(recs)} scenes, dim={H[next(iter(H))].shape}")
    out = {"run": args.run, "n_scenes": len(recs),
           "behaviour": behaviour(recs),
           "probe": probe_block(recs, H),
           "law": law_block(recs, H, args.n_boot),
           "sdt": sdt_block(recs, min(args.n_boot, 1000)),
           "gate": gate_block(recs, H, min(args.n_boot, 1000))}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
