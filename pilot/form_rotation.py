"""What carries the evidence-axis rotation across prompt forms?

All statistics are computed within a form, from final hidden states.

- reliability: split-half cosine of the evidence axis v0 and of the conflict
  axis va (same form, same cells, disjoint scenes).  This is the ceiling for
  any content-driven axis change.
- cos(va, v_b): the two conflict axes differ only in the colour word of the
  hint (a vs b).  A cosine below the split-half ceiling means the hint's
  colour content, not the extra sentence, moves the axis.
- cos(rot_a, h): rotation of va away from v0, projected on the colour-swap
  direction h = mean(absent b) - mean(absent a) measured on held-out scenes.
- hint decoding: 11-way colour readout of the stated colour from final
  states, per form and transferred from the standard form.
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CANDIDATES, DISTRACTOR

CELLS = (("absent", "none"), ("absent", "a"), ("absent", "b"),
         ("sufficient", "none"), ("sufficient", "wrong"))


def load(path):
    z = np.load(path, allow_pickle=True)
    idx, scenes = defaultdict(list), defaultdict(list)
    for i, key in enumerate(str(k) for k in z["keys"]):
        s, form, ev, hint = key.split("|")
        idx[(form, ev, hint)].append(i)
        scenes[(form, ev, hint)].append(s)
    return z["H"].astype(np.float32), idx, scenes


def unit(v):
    return v / np.linalg.norm(v)


def means(H, idx, form, sel):
    return {c: H[np.array(idx[(form,) + c])[sel]].mean(0) for c in CELLS}


def geometry(m):
    v0 = m[("sufficient", "none")] - m[("absent", "none")]
    va = m[("sufficient", "wrong")] - m[("absent", "a")]
    vb = m[("sufficient", "wrong")] - m[("absent", "b")]
    h = m[("absent", "b")] - m[("absent", "a")]
    return v0, va, vb, h


def split_half(H, idx, form, reps=100, seed=0):
    n = len(idx[(form, "absent", "none")])
    rng = np.random.default_rng(seed)
    cos0, cosa, cosab, cosrot = [], [], [], []
    for _ in range(reps):
        p = rng.permutation(n)
        h1, h2 = p[: n // 2], p[n // 2:]
        v0a, vaa, vba, ha = geometry(means(H, idx, form, h1))
        v0b, vab, vbb, hb = geometry(means(H, idx, form, h2))
        cos0.append(unit(v0a) @ unit(v0b))
        cosa.append(unit(vaa) @ unit(vab))
        cosab.append(unit(vaa) @ unit(vba))
        u0 = unit((v0a + v0b) / 2)
        rot = unit(vaa - (vaa @ u0) * u0)
        cosrot.append(rot @ unit(hb))
    return {k: np.array(v) for k, v in
            dict(cos_v0=cos0, cos_va=cosa, cos_va_vb=cosab,
                 cos_rot_h=cosrot).items()}


def colour_of(row, cell):
    if cell == ("absent", "a"):
        return row["absent_hint"]
    if cell == ("absent", "b"):
        return DISTRACTOR[row["absent_hint"]]
    return DISTRACTOR[row["gold"]]


def decode(H, idx, rows, form, cell, train=None):
    ii = np.array(idx[(form,) + cell])
    y = np.array([CANDIDATES.index(colour_of(r, cell)) for r in rows])
    X = H[ii]
    lr = dict(C=0.05, max_iter=500)
    if train is None:
        groups = np.array([r["sample_id"] for r in rows])
        pred = np.zeros(len(y), dtype=int)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(**lr))
            pipe.fit(X[tr], y[tr])
            pred[te] = pipe.predict(X[te])
        return float(accuracy_score(y, pred))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(**lr))
    pipe.fit(H[np.array(idx[(train,) + cell])],
             np.array([CANDIDATES.index(colour_of(r, cell)) for r in rows]))
    return float(accuracy_score(y, pipe.predict(X)))


def main(path, rows_path, ref="std", mode="geo"):
    H, idx, scenes = load(path)
    rows = [r for r in (json.loads(line) for line in open(rows_path))
            if r.get("probs")]
    forms = sorted({k[0] for k in idx})
    name = path.split("/")[-1]
    print(f"\n## {name}  n={len(rows)}")
    counts = Counter(r["absent_hint"] for r in rows)
    print("hint colours:", dict(sorted(counts.items())))
    if mode in ("geo", "all"):
        print("\n| form | cos_v0 split-half | cos_va split-half | cos(va,vb) | cos(rot_a,h) held-out |")
        print("|---|---|---|---|---|")
        table = {}
        for form in forms:
            s = split_half(H, idx, form)
            row = {k: (float(np.mean(v)), [float(x) for x in
                                           np.percentile(v, [2.5, 97.5])])
                   for k, v in s.items()}
            table[form] = row
            print(f"| {form} | {row['cos_v0'][0]:.3f} "
                  f"[{row['cos_v0'][1][0]:.3f},{row['cos_v0'][1][1]:.3f}] "
                  f"| {row['cos_va'][0]:.3f} "
                  f"[{row['cos_va'][1][0]:.3f},{row['cos_va'][1][1]:.3f}] "
                  f"| {row['cos_va_vb'][0]:.3f} "
                  f"[{row['cos_va_vb'][1][0]:.3f},{row['cos_va_vb'][1][1]:.3f}] "
                  f"| {row['cos_rot_h'][0]:+.3f} "
                  f"[{row['cos_rot_h'][1][0]:+.3f},{row['cos_rot_h'][1][1]:+.3f}] |",
                  flush=True)
        json.dump(table, open(path.replace(".npz", "_rotation.json"), "w"),
                  indent=1)
    if mode in ("decode", "all"):
        print("\n| form | hint acc abs-a (CV) | hint acc suf-wrong (CV) | "
              "transfer std->form (abs-a) |", flush=True)
        print("|---|---|---|---|", flush=True)
        for form in forms:
            a = decode(H, idx, rows, form, ("absent", "a"))
            w = decode(H, idx, rows, form, ("sufficient", "wrong"))
            t = decode(H, idx, rows, form, ("absent", "a"), train=ref)
            print(f"| {form} | {a:.3f} | {w:.3f} | {t:.3f} |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "std",
         sys.argv[4] if len(sys.argv) > 4 else "all")
