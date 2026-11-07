"""Is the evidence axis compressed or rotated by the hint?

The paper's headline gap is a projection onto the no-hint direction w, so it
cannot separate a shorter evidence vector from one that has turned away from
w. This computes, per condition, the full mean-difference vector, its norm,
its norm in units of the no-hint within-class noise (Mahalanobis), its angle
to w, and the per-scene coherence of the paired gaps. It also reports the
frozen transfer honestly: the probe is fitted on training scenes' no-hint
states and evaluated on held-out scenes' hinted states.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc

PAIRS = {
    "none": ("sufficient|none", "absent|none"),
    "correct": ("sufficient|correct", "absent|a"),
    "conflict": ("sufficient|wrong", "absent|a"),
    "conflict_b": ("sufficient|wrong", "absent|b"),
}


def load(run_path):
    lines = [json.loads(l) for l in open(run_path)]
    recs = [r for r in lines if "meta" not in r]
    z = np.load(run_path.replace(".jsonl", ".npz"), allow_pickle=False)
    H = dict(zip(z["keys"].tolist(), z["H"]))
    return recs, H


def stack(recs, H, key):
    X, sid = [], []
    for r in recs:
        k = f"{r['sample_id']}|{key}"
        X.append(H[k].astype(np.float32))
        sid.append(r["sample_id"])
    return np.stack(X), np.array(sid)


def whitener(X, y, ridge=0.01):
    """Within-class covariance whitening, built from the no-hint pairs."""
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    Z = np.where((y == 1)[:, None], X - m1, X - m0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    lam = ridge * float(np.mean(S ** 2))
    return Vt.T @ np.diag(1.0 / np.sqrt(S ** 2 + lam))


def mean_gap(Xp, Xa, W, w):
    d = Xp.mean(0) - Xa.mean(0)
    dw = d @ W
    return {
        "euclid": float(np.linalg.norm(d)),
        "mahal": float(np.linalg.norm(dw)),
        "along_w": float(d @ w),
        "cos_w": float((d @ w) / (np.linalg.norm(d) + 1e-12)),
    }


def paired(v):
    """Per-scene gap vectors v_s: how much of their energy shares a direction."""
    norms = np.linalg.norm(v, axis=1)
    mean_vec = v.mean(0)
    return {
        "mean_norm": float(norms.mean()),
        "rms_norm": float(np.sqrt((norms ** 2).mean())),
        "coherence": float(np.linalg.norm(mean_vec) / np.sqrt((norms ** 2).mean())),
        "cos_to_mean": float(np.mean(v @ mean_vec / (norms * np.linalg.norm(mean_vec) + 1e-12))),
    }


def boot(fn, n, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    arr = np.stack([np.asarray([v[k] for k in vals[0]]) for v in vals])
    keys = list(vals[0])
    return {k: [float(np.mean([v[k] for v in vals])),
                float(np.quantile(arr[:, i], 0.025)),
                float(np.quantile(arr[:, i], 0.975))]
            for i, k in enumerate(keys)}


def transfer(recs, H, C, tune, cond="conflict"):
    Xn, _ = stack(recs, H, PAIRS["none"][0])
    An, _ = stack(recs, H, PAIRS["none"][1])
    Xc, _ = stack(recs, H, PAIRS[cond][0])
    Ac, _ = stack(recs, H, PAIRS[cond][1])
    sid = np.array([r["sample_id"] for r in recs])
    X0 = np.concatenate([Xn, An])
    y0 = np.concatenate([np.ones(len(Xn)), np.zeros(len(An))])
    g0 = np.concatenate([sid, sid])
    X1 = np.concatenate([Xc, Ac])
    y1 = np.concatenate([np.ones(len(Xc)), np.zeros(len(Ac))])

    def fit(X, y, c):
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=c, max_iter=5000)).fit(X, y)

    Cs = [0.0003, 0.001, 0.003, 0.01, 0.1] if tune else [C]

    # leaky: fitted on every no-hint scene, scored on the same scenes' conflict
    leaky = 0.0
    best_h = (0.0, C)
    for c in Cs:
        s = fit(X0, y0, c).decision_function(X1)
        v = auroc(s, y1)
        if v > leaky:
            leaky = v
        best_h = (v, c)
    leaky_c = best_h[1]

    # honest: held-out scenes, train on no-hint, score on conflict
    oof = np.zeros(len(y1))
    for tr, te in GroupKFold(n_splits=5).split(X0, y0, g0):
        te_idx = set(g0[te])
        m = np.array([s in te_idx for s in sid])
        ev = np.concatenate([m, m])
        c = leaky_c
        clf = fit(X0[~ev], y0[~ev], c)
        oof[ev] = clf.decision_function(X1[ev])
    mask = oof != 0
    held = auroc(oof[mask], y1[mask])

    # refit in-domain on conflict, still scene-held-out
    oof2 = np.zeros(len(y1))
    for tr, te in GroupKFold(n_splits=5).split(X1, y1, np.concatenate([sid, sid])):
        clf = fit(X1[tr], y1[tr], leaky_c)
        oof2[te] = clf.decision_function(X1[te])
    refit = auroc(oof2, y1)
    return {"leaky": float(leaky), "heldout": float(held),
            "refit": float(refit), "C": float(leaky_c)}, y1, mask, oof, np.concatenate([sid, sid])


def run(run_path, C):
    recs, H = load(run_path)
    P, sid = stack(recs, H, PAIRS["none"][0])
    A, _ = stack(recs, H, PAIRS["none"][1])
    X = np.concatenate([P, A])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(A))])
    W = whitener(X, y)
    d0 = P.mean(0) - A.mean(0)
    w = d0 / (np.linalg.norm(d0) + 1e-12)
    n = len(P)

    out = {"n": int(n), "dim": int(P.shape[1]),
           "gap": {}, "paired": {}, "transfer": {}}
    for name, (pk, nk) in PAIRS.items():
        Xp, _ = stack(recs, H, pk)
        Xa, _ = stack(recs, H, nk)
        out["gap"][name] = mean_gap(Xp, Xa, W, w)
        out["paired"][name] = paired(Xp - Xa)
        if name == "none":
            pass

    def mk(name):
        Xp, _ = stack(recs, H, PAIRS[name][0])
        Xa, _ = stack(recs, H, PAIRS[name][1])
        return Xp, Xa

    def boot_gap(name):
        Xp, Xa = mk(name)

        def f(idx):
            return mean_gap(Xp[idx], Xa[idx], W, w)
        return boot(f, n)

    def boot_paired(name):
        Xp, Xa = mk(name)

        def f(idx):
            return paired((Xp - Xa)[idx])
        return boot(f, n)

    out["gap_ci"] = {k: boot_gap(k) for k in PAIRS}
    out["paired_ci"] = {k: boot_paired(k) for k in PAIRS}

    def boot_rel(name):
        Xp, Xa = mk(name)

        def f(idx):
            d = Xp[idx].mean(0) - Xa[idx].mean(0)
            return {"ratio": float(np.linalg.norm(d) / np.linalg.norm(d0)),
                    "angle": float(np.degrees(np.arccos(np.clip(d @ w / np.linalg.norm(d), -1, 1))))}
        return boot(f, n)

    out["rel"] = {k: boot_rel(k) for k in PAIRS}

    dirs = {}
    for name, (pk, nk) in PAIRS.items():
        Xp, _ = stack(recs, H, pk)
        Xa, _ = stack(recs, H, nk)
        d = Xp.mean(0) - Xa.mean(0)
        dirs[name] = d / (np.linalg.norm(d) + 1e-12)
    names = list(PAIRS)
    out["cos_between"] = {f"{a}/{b}": float(dirs[a] @ dirs[b])
                          for i, a in enumerate(names) for b in names[i + 1:]}

    for tag, tune in (("fixed", False), ("tuned", True)):
        tr, y1, mask, oof, sid1 = transfer(recs, H, C, tune)
        # grouped CI on the held-out scores
        uniq = np.unique(sid1[mask])
        idx_of = {k: np.where(sid1 == k)[0] for k in uniq}
        idx_of = {k: v[mask[v]] for k, v in idx_of.items() if mask[v].any()}
        keys = list(idx_of)
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(1000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = np.concatenate([idx_of[keys[q]] for q in pick])
            vals.append(auroc(oof[idx], y1[idx]))
        tr["heldout_ci"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
        out["transfer"][tag] = tr

    out["transfer_cond"] = {}
    for cond in ("correct", "conflict", "conflict_b"):
        tr, y1, mask, oof, sid1 = transfer(recs, H, C, False, cond)
        uniq = np.unique(sid1[mask])
        idx_of = {k: np.where(sid1 == k)[0] for k in uniq}
        idx_of = {k: v[mask[v]] for k, v in idx_of.items() if mask[v].any()}
        keys = list(idx_of)
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(1000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = np.concatenate([idx_of[keys[q]] for q in pick])
            vals.append(auroc(oof[idx], y1[idx]))
        tr["heldout_ci"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
        out["transfer_cond"][cond] = tr

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["runs/v5_261.jsonl"])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--out", default="results/geometry.json")
    args = ap.parse_args()

    res = {}
    for r in args.run:
        tag = r.split("/")[-1].replace(".jsonl", "")
        res[tag] = run(r, args.C)
        o = res[tag]
        print(f"\n{tag}  n={o['n']}  dim={o['dim']}")
        for name in PAIRS:
            g, gc = o["gap"][name], o["gap_ci"][name]
            p, pc = o["paired"][name], o["paired_ci"][name]
            print(f"  {name:10s} |dmu| {g['euclid']:7.2f} "
                  f"[{gc['euclid'][1]:.2f},{gc['euclid'][2]:.2f}]  "
                  f"mahal {g['mahal']:6.2f} [{gc['mahal'][1]:.2f},{gc['mahal'][2]:.2f}]  "
                  f"along_w {g['along_w']:7.2f} [{gc['along_w'][1]:.2f},{gc['along_w'][2]:.2f}]  "
                  f"cos_w {g['cos_w']:+.3f}")
            print(f"             per-scene |v| {p['mean_norm']:6.2f} "
                  f"[{pc['mean_norm'][1]:.2f},{pc['mean_norm'][2]:.2f}]  "
                  f"coherence {p['coherence']:.3f} "
                  f"[{pc['coherence'][1]:.3f},{pc['coherence'][2]:.3f}]  "
                  f"ratio {o['rel'][name]['ratio'][0]:.3f} "
                  f"[{o['rel'][name]['ratio'][1]:.3f},{o['rel'][name]['ratio'][2]:.3f}]  "
                  f"angle {o['rel'][name]['angle'][0]:.1f} "
                  f"[{o['rel'][name]['angle'][1]:.1f},{o['rel'][name]['angle'][2]:.1f}]")
        for tag2, t in o["transfer"].items():
            print(f"  transfer {tag2}: leaky {t['leaky']:.3f}  held-out {t['heldout']:.3f} "
                  f"[{t['heldout_ci'][0]:.3f},{t['heldout_ci'][1]:.3f}]  "
                  f"refit {t['refit']:.3f}  C={t['C']:g}")
        for cond, t in o["transfer_cond"].items():
            print(f"  held-out {cond:10s}: leaky {t['leaky']:.3f}  held-out {t['heldout']:.3f} "
                  f"[{t['heldout_ci'][0]:.3f},{t['heldout_ci'][1]:.3f}]  refit {t['refit']:.3f}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
