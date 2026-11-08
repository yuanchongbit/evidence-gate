"""Evidence-axis geometry across prompt forms, from saved final states."""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(path):
    z = np.load(path, allow_pickle=True)
    idx, scenes = defaultdict(list), defaultdict(list)
    for i, k in enumerate(str(x) for x in z["keys"]):
        s, form, ev, hint = k.split("|")
        idx[(form, ev, hint)].append(i)
        scenes[(form, ev, hint)].append(s)
    return z["H"].astype(np.float32), idx, scenes


def geometry(H, idx):
    out = {}
    forms = sorted({k[0] for k in idx})
    for form in forms:
        def mat(ev, hint):
            return H[idx[(form, ev, hint)]]

        v0 = mat("sufficient", "none").mean(0) - mat("absent", "none").mean(0)
        va = mat("sufficient", "wrong").mean(0) - mat("absent", "a").mean(0)
        vb = mat("sufficient", "wrong").mean(0) - mat("absent", "b").mean(0)
        n0, na, nb = np.linalg.norm(v0), np.linalg.norm(va), np.linalg.norm(vb)
        cos0 = float(va @ v0 / (na * n0))
        out[form] = {
            "length_ratio": float(na / n0),
            "angle_deg": float(np.degrees(np.arccos(np.clip(cos0, -1, 1)))),
            "cos_hint_pairs": float(va @ vb / (na * nb)),
            "frozen_probe_conflict": frozen_auc(H, idx, form),
        }
    return out


def frozen_auc(H, idx, form):
    X = np.vstack([H[idx[(form, "sufficient", "none")]], H[idx[(form, "absent", "none")]]])
    y = np.r_[np.ones(len(idx[(form, "sufficient", "none")])),
              np.zeros(len(idx[(form, "absent", "none")]))]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    pipe.fit(X, y)
    Xc = np.vstack([H[idx[(form, "sufficient", "wrong")]], H[idx[(form, "absent", "a")]]])
    yc = np.r_[np.ones(len(idx[(form, "sufficient", "wrong")])),
               np.zeros(len(idx[(form, "absent", "a")]))]
    return float(roc_auc_score(yc, pipe.decision_function(Xc)))


def main(src, dst):
    H, idx, _ = load(src)
    json.dump(geometry(H, idx), open(dst, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
