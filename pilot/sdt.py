"""Signal-detection / psychometric analysis of the refusal decision.

Question: when a conflicting hint lowers refusal, does it move the decision
criterion along a fixed evidence axis, or does it change the evidence axis
itself?

We use the no-hint probe score as the evidence axis. The probe is fitted with
GroupKFold on scenes and applied out-of-fold to every hint condition, so a
scene never contributes to its own score. We then fit

    logit P(refuse) = b0 + a*z + sum_k d_k*(b_k + a_k*z)

where z is the probe score standardised on the no-hint pair and k indexes the
hint conditions. A pure criterion shift gives b_k != 0 with a_k = 0; a change
in the evidence representation gives a_k != 0. CIs come from a cluster
bootstrap over scenes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc
from .probe import dataset, load

ROOT = Path(__file__).resolve().parent.parent

# condition -> (present key, absent keys). The correct-hint condition changes
# only the present prompt; there is no correct hint for an absent object.
CONDITIONS = {
    "none": ("sufficient|none", ["absent|none"]),
    "conflict": ("sufficient|wrong", ["absent|a", "absent|b"]),
    "correct": ("sufficient|correct", ["absent|none"]),
}
ORDER = ["none", "conflict", "correct"]


def choose_C(recs, H, Cs, group_by="sample"):
    pos_key, neg_keys = CONDITIONS["none"]
    X, y, g, _, _ = dataset(recs, H, pos_key, neg_keys[0], group_by=group_by)
    best = (-1.0, None)
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.decision_function(X[te])
        score = auroc(oof, y)
        if score > best[0]:
            best = (score, C)
    return best[1], best[0]


def oof_scores(recs, H, C):
    """Out-of-fold probe score for every (scene, condition) state."""
    pos_key, neg_keys = CONDITIONS["none"]
    X, y, g, _, _ = dataset(recs, H, pos_key, neg_keys[0], group_by="sample")
    row_recs = [r for r in recs for key in (pos_key, neg_keys[0])
                if f"{r['sample_id']}|{key}" in H]
    out = {}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        for i in te:
            r = row_recs[i]
            sid = r["sample_id"]
            for key in {k for _, keys in CONDITIONS.values() for k in keys} | {
                CONDITIONS[c][0] for c in CONDITIONS
            }:
                hk = f"{sid}|{key}"
                if hk in H:
                    out[(sid, key)] = float(clf.decision_function(H[hk][None, :])[0])
    return out


def rows_for(recs, scores, conditions):
    rows = []
    for r in recs:
        sid = r["sample_id"]
        for cond in conditions:
            pos_key, neg_keys = CONDITIONS[cond]
            entries = [(pos_key, 0)]
            entries += [(k, 1) for k in neg_keys]
            for key, absent in entries:
                if (sid, key) not in scores:
                    continue
                p = r["probs"][key]
                rows.append({
                    "sample_id": sid,
                    "cond": cond,
                    "absent": absent,
                    "score": scores[(sid, key)],
                    "refuse": int(np.argmax(p) == len(p) - 1),
                })
    return rows


def design(rows, conditions):
    z = np.array([r["score"] for r in rows], float)
    n = len(rows)
    cols = [np.ones(n), z]
    names = ["b0", "a"]
    for cond in conditions[1:]:
        d = np.array([r["cond"] == cond for r in rows], float)
        cols += [d, d * z]
        names += [f"b_{cond}", f"a_{cond}"]
    return np.column_stack(cols), np.array([r["refuse"] for r in rows], float), names


def fit_logit(X, y, lam=1e-4):
    def nll(beta):
        eta = X @ beta
        # stable log-likelihood
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))
        return -ll + 0.5 * lam * np.sum(beta[1:] ** 2)

    def grad(beta):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        g = X.T @ (p - y)
        g[1:] += lam * beta[1:]
        return g

    res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12})
    return res.x


def loglik(beta, X, y):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_summary(rows, conditions):
    X, y, names = design(rows, conditions)
    beta = fit_logit(X, y)
    ll_full = loglik(beta, X, y)
    k = X.shape[1]
    a0 = beta[1]
    out = {
        "names": names,
        "beta": {n: float(b) for n, b in zip(names, beta)},
        "loglik": ll_full,
        "aic": float(2 * k - 2 * ll_full),
    }
    # parallel model: drop the interactions
    keep = [i for i, n in enumerate(names) if not n.startswith("a_")]
    beta_p = fit_logit(X[:, keep], y)
    ll_p = loglik(beta_p, X[:, keep], y)
    lr = 2 * (ll_full - ll_p)
    df = len(names) - len(keep)
    from scipy.stats import chi2
    out["parallel"] = {
        "beta": {n: float(b) for n, b in zip([names[i] for i in keep], beta_p)},
        "loglik": ll_p,
        "aic": float(2 * len(keep) - 2 * ll_p),
        "lr_stat": float(lr),
        "df": int(df),
        "p": float(chi2.sf(lr, df)),
    }
    # shift in z units and ED50 (z where P(refuse)=0.5)
    shifts = {}
    for cond in conditions[1:]:
        b = beta[names.index(f"b_{cond}")]
        a = beta[names.index(f"a_{cond}")]
        b0 = beta[names.index("b0")]
        shifts[cond] = {
            "delta_b": float(b),
            "delta_a": float(a),
            "ed50_none": float(-b0 / a0),
            "ed50_cond": float(-(b0 + b) / (a0 + a)),
            "horizontal_shift_at_mean_slope": float(-b / ((a0 + a) / 2)),
        }
    out["shifts"] = shifts
    return out, X, y, names, beta


def cluster_bootstrap(rows, conditions, n_boot=2000, seed=0):
    ids = sorted({r["sample_id"] for r in rows})
    by_id = {i: [r for r in rows if r["sample_id"] == i] for i in ids}
    rng = np.random.default_rng(seed)
    vals = {c: {"delta_b": [], "delta_a": [], "ed50": [], "ed50_diff": [],
                "shift": []} for c in conditions[1:]}
    bin_vals = {c: [] for c in conditions}
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sample = [r for i in pick for r in by_id[i]]
        X, y, names = design(sample, conditions)
        beta = fit_logit(X, y)
        a0 = beta[1]
        ed50_none = -beta[0] / a0
        for c in conditions[1:]:
            b = beta[names.index(f"b_{c}")]
            a = beta[names.index(f"a_{c}")]
            b0 = beta[0]
            ed50 = -(b0 + b) / (a0 + a)
            vals[c]["delta_b"].append(b)
            vals[c]["delta_a"].append(a)
            vals[c]["ed50"].append(ed50)
            vals[c]["ed50_diff"].append(ed50 - ed50_none)
            vals[c]["shift"].append(-b / ((a0 + a) / 2))
        for c in conditions:
            sub = [r["refuse"] for r in sample if r["cond"] == c]
            bin_vals[c].append(float(np.mean(sub)))
    out = {}
    for c in conditions[1:]:
        out[c] = {}
        for k, v in vals[c].items():
            v = np.array(v)
            out[c][k] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return out


def binary_sdt(recs, conditions):
    out = {}
    for c in conditions:
        pos_key, neg_keys = CONDITIONS[c]
        pres = [r["probs"][pos_key] for r in recs if pos_key in r["probs"]]
        neg = [r["probs"][k] for r in recs for k in neg_keys if k in r["probs"]]
        fa = np.mean([np.argmax(p) == len(p) - 1 for p in pres])
        hit = np.mean([np.argmax(p) == len(p) - 1 for p in neg])
        # log-linear correction keeps d' finite at 0/1 rates
        n1, n2 = len(pres), len(neg)
        fa_c = (fa * n1 + 0.5) / (n1 + 1.0)
        hit_c = (hit * n2 + 0.5) / (n2 + 1.0)
        out[c] = {
            "n_present": n1, "n_absent": n2,
            "fa_refuse": float(fa), "hit_refuse": float(hit),
            "dprime": float(norm.ppf(hit_c) - norm.ppf(fa_c)),
            "criterion": float(-0.5 * (norm.ppf(hit_c) + norm.ppf(fa_c))),
        }
    return out


def binary_bootstrap(recs, conditions, n_boot=2000, seed=0):
    """Cluster bootstrap over scenes for d', c and their paired shifts."""
    ids = sorted({r["sample_id"] for r in recs})
    by_id = {i: [r for r in recs if r["sample_id"] == i] for i in ids}
    rng = np.random.default_rng(seed)
    keys = ["dprime", "criterion"]
    vals = {c: {k: [] for k in keys} for c in conditions}
    for c in conditions[1:]:
        for k in keys:
            vals[c][f"{k}_diff"] = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sample = [r for i in pick for r in by_id[i]]
        s = binary_sdt(sample, conditions)
        for c in conditions:
            for k in keys:
                vals[c][k].append(s[c][k])
        for c in conditions[1:]:
            for k in keys:
                vals[c][f"{k}_diff"].append(s[c][k] - s["none"][k])
    return {
        c: {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in d.items()}
        for c, d in vals.items()
    }


def binned(rows, conditions, nbins=8):
    z = np.array([r["score"] for r in rows])
    edges = np.quantile(z, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for c in conditions:
        for j in range(nbins):
            sub = [r for r in rows if r["cond"] == c and edges[j] <= r["score"] < edges[j + 1]]
            if not sub:
                continue
            p = float(np.mean([r["refuse"] for r in sub]))
            out.append({
                "cond": c,
                "x": float(np.mean([r["score"] for r in sub])),
                "p": p,
                "se": float(np.sqrt(max(p * (1 - p), 1e-6) / len(sub))),
                "n": len(sub),
            })
    return out


def plot(rows, fit, bins, conditions, path, n_boot=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.array([r["score"] for r in rows])
    grid = np.linspace(np.quantile(z, 0.005), np.quantile(z, 0.995), 300)
    colors = {"none": "#1b2a41", "conflict": "#c0392b", "correct": "#2e7d32"}
    labels = {"none": "no hint", "conflict": "conflicting hint", "correct": "correct hint"}

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for c in conditions:
        b = bins_of(bins, c)
        ax.errorbar([r["x"] for r in b], [r["p"] for r in b], yerr=[r["se"] for r in b],
                    fmt="o", ms=3.2, lw=1, capsize=1.6, color=colors[c],
                    mfc="white", mec=colors[c], zorder=3)
        beta = fit["beta"]
        names = fit["names"]
        eta = beta["b0"] + beta["a"] * grid
        if c != "none":
            eta = eta + beta[f"b_{c}"] + beta[f"a_{c}"] * grid
        ax.plot(grid, 1 / (1 + np.exp(-eta)), color=colors[c], lw=1.4,
                label=labels[c], zorder=2)
    ax.axhline(0.5, color="0.8", lw=0.6, zorder=1)
    ax.set_xlabel("probe evidence score (no-hint standardised)")
    ax.set_ylabel("P(refuse)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bins_of(bins, c):
    return [r for r in bins if r["cond"] == c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--conditions", default="none,conflict,correct")
    ap.add_argument("--Cs", default="0.1",
                    help="probe regularisation; fixed to the main probe's C")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "sdt.json"))
    ap.add_argument("--fig", default=str(ROOT / "paper" / "figures" / "fig_sdt.pdf"))
    args = ap.parse_args()

    recs, _, H = load(args.run)
    conditions = [c for c in ORDER if c in args.conditions.split(",")]
    C, auroc_none = choose_C(recs, H, [float(x) for x in args.Cs.split(",")])
    scores = oof_scores(recs, H, C)
    rows = rows_for(recs, scores, conditions)
    print(f"n={len(recs)} scenes, C={C:g}, no-hint AUROC={auroc_none:.3f}, "
          f"{len(rows)} item-conditions")

    fit, X, y, names, beta = fit_summary(rows, conditions)
    print("full fit:", " ".join(f"{n}={b:+.3f}" for n, b in fit["beta"].items()))
    print(f"AIC full {fit['aic']:.1f}  parallel {fit['parallel']['aic']:.1f}  "
          f"LR {fit['parallel']['lr_stat']:.1f} df={fit['parallel']['df']} "
          f"p={fit['parallel']['p']:.3g}")
    for c in conditions[1:]:
        s = fit["shifts"][c]
        print(f"{c}: delta_b={s['delta_b']:+.3f} delta_a={s['delta_a']:+.3f} "
              f"ED50 {s['ed50_none']:+.3f} -> {s['ed50_cond']:+.3f} "
              f"shift={s['horizontal_shift_at_mean_slope']:+.3f}")

    ci = cluster_bootstrap(rows, conditions, n_boot=args.n_boot)
    for c in conditions[1:]:
        print(f"{c} bootstrap 95% CI: delta_b {ci[c]['delta_b']}  "
              f"delta_a {ci[c]['delta_a']}  ed50_diff {ci[c]['ed50_diff']}")

    sdt = binary_sdt(recs, conditions)
    for c in conditions:
        print(f"{c}: d'={sdt[c]['dprime']:.2f} c={sdt[c]['criterion']:+.2f} "
              f"(refuse present {sdt[c]['fa_refuse']:.3f}, absent {sdt[c]['hit_refuse']:.3f})")
    sdt_ci = binary_bootstrap(recs, conditions, n_boot=args.n_boot)
    for c in conditions[1:]:
        print(f"{c} binary 95% CI: d' {sdt_ci[c]['dprime']} "
              f"delta_d' {sdt_ci[c]['dprime_diff']} c {sdt_ci[c]['criterion']} "
              f"delta_c {sdt_ci[c]['criterion_diff']}")

    bins = binned(rows, conditions)
    plot(rows, fit, bins, conditions, args.fig, n_boot=args.n_boot)
    print("figure ->", args.fig)

    payload = {
        "run": args.run, "n_scenes": len(recs), "C": C,
        "auroc_no_hint": auroc_none, "conditions": conditions,
        "fit": fit, "bootstrap_ci": ci, "binary_sdt": sdt,
        "binary_ci": sdt_ci, "bins": bins,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
