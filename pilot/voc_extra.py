"""VOC extras the main block does not cover: frozen transfer, rotation, coherence.

Reuses the already-uploaded voc_analyze module for loading and metrics so the
protocol matches the COCO analysis (scene-grouped folds, C=0.1, fixed axis).
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pilot import voc_analyze as VA
from pilot.geometry import PAIRS, load, stack
from pilot.metrics import auroc

CONDS = [("correct", PAIRS["correct"]), ("conflict", PAIRS["conflict"]),
         ("conflict_b", PAIRS["conflict_b"])]


def mean_vec(recs, H, key):
    X, _ = stack(recs, H, key)
    return X.mean(0)


def pair_vec(recs, H, pos, neg):
    return mean_vec(recs, H, pos) - mean_vec(recs, H, neg)


def angle(u, v):
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def coherence(recs, H, pos, neg, axis):
    """Mean cosine between each scene's own gap vector and the reference axis."""
    out = []
    for r in recs:
        kp = f"{r['sample_id']}|{pos}"
        kn = f"{r['sample_id']}|{neg}"
        if kp not in H or kn not in H:
            continue
        v = H[kp].astype(np.float32) - H[kn].astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            out.append(float(v @ axis / n))
    return float(np.mean(out))


def frozen(recs, H):
    """Train on the no-hint pair, score hinted states of held-out scenes."""
    Xn, yn, gn = VA.probe_dataset(recs, H, *PAIRS["none"])
    out = {}
    for name, (pk, nk) in CONDS:
        Xh, yh, gh = VA.probe_dataset(recs, H, pk, nk)
        oof = np.zeros(len(yh))
        for tr, te in GroupKFold(n_splits=5).split(Xn, yn, gn):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.1, max_iter=3000))
            clf.fit(Xn[tr], yn[tr])
            held = np.isin(gh, np.unique(gn[te]))
            if held.any():
                oof[held] = clf.decision_function(Xh[held])
        out[name] = {"auroc": float(auroc(oof, yh)),
                     "ci": VA.scene_boot_auroc(oof, yh, gh, n_boot=500)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/tmp/voc_3b.jsonl")
    ap.add_argument("--out", default="/tmp/voc_extra.json")
    args = ap.parse_args()
    recs, H = load(args.run)
    d0 = pair_vec(recs, H, *PAIRS["none"])
    w = d0 / np.linalg.norm(d0)

    point = {"angle_vs_none": {}, "gap_ratio": {}, "coherence": {}}
    point["coherence"]["none"] = coherence(recs, H, *PAIRS["none"], w)
    for name, (pk, nk) in CONDS:
        d = pair_vec(recs, H, pk, nk)
        point["angle_vs_none"][name] = angle(d, d0)
        point["gap_ratio"][name] = float(d @ w / (d0 @ w))
        point["coherence"][name] = coherence(recs, H, pk, nk, w)
    dc = pair_vec(recs, H, *PAIRS["conflict"])
    db = pair_vec(recs, H, *PAIRS["conflict_b"])
    point["cos_conflict_ab"] = float(dc @ db / (np.linalg.norm(dc) * np.linalg.norm(db)))

    ids = np.array([r["sample_id"] for r in recs])
    uniq = np.unique(ids)
    idx_of = {k: np.where(ids == k)[0] for k in uniq}
    rng = np.random.default_rng(0)
    draws = {"angle_vs_none": {n: [] for n, _ in CONDS},
             "gap_ratio": {n: [] for n, _ in CONDS},
             "coherence": {n: [] for n, _ in CONDS},
             "cos_conflict_ab": []}
    for _ in range(500):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rs = [recs[i] for k in pick for i in idx_of[k]]
        try:
            e0 = pair_vec(rs, H, *PAIRS["none"])
            ww = e0 / np.linalg.norm(e0)
            for name, (pk, nk) in CONDS:
                d = pair_vec(rs, H, pk, nk)
                draws["angle_vs_none"][name].append(angle(d, e0))
                draws["gap_ratio"][name].append(float(d @ ww / (e0 @ ww)))
                draws["coherence"][name].append(coherence(rs, H, pk, nk, ww))
            ec = pair_vec(rs, H, *PAIRS["conflict"])
            eb = pair_vec(rs, H, *PAIRS["conflict_b"])
            draws["cos_conflict_ab"].append(
                float(ec @ eb / (np.linalg.norm(ec) * np.linalg.norm(eb))))
        except Exception:
            continue

    def ci(v):
        v = np.array(v, float)
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    out = {"point": point,
           "ci": {k: ({n: ci(v) for n, v in d.items()} if isinstance(d, dict)
                      else ci(d)) for k, d in draws.items()},
           "frozen": frozen(recs, H)}
    cells = [(point["gap_ratio"][n], VA.law_cells(recs, H)[n]["dp_ratio"])
             for n, _ in CONDS]
    g = np.array([c[0] for c in cells])
    d = np.array([c[1] for c in cells])
    slope0 = float((g * d).sum() / (g * g).sum())
    out["law_voc_only"] = {"cells": cells, "through_origin_slope": slope0,
                           "rmse_vs_1.425": float(np.sqrt(((d - 1.425 * g) ** 2).mean()))}
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
