"""Cross-cell decoding of the stated colour.

Within a scene the absent-arm cells differ only in the colour word: hint a
states acolor, hint b states DISTRACTOR[acolor].  A decoder trained to read
acolor from a-states and tested on b-states must output a *different* class for
the same scene, so it can only succeed by reading the stated colour rather than
the category or the image.  This separates "the hint is represented" from
"the question category is represented".
"""

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CANDIDATES, DISTRACTOR
from .form_rotation import load


def labels(rows, cell):
    if cell == "a":
        return [r["absent_hint"] for r in rows]
    return [DISTRACTOR[r["absent_hint"]] for r in rows]


def fit(H, idx, rows, form, cell):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=500,
                                            class_weight="balanced"))
    y = np.array([CANDIDATES.index(c) for c in labels(rows, cell)])
    pipe.fit(H[np.array(idx[(form, "absent", cell)])], y)
    return pipe


def acc(pipe, H, idx, rows, form, cell):
    y = np.array([CANDIDATES.index(c) for c in labels(rows, cell)])
    X = H[np.array(idx[(form, "absent", cell)])]
    pred = pipe.predict(X)
    return (float(accuracy_score(y, pred)),
            float(balanced_accuracy_score(y, pred)),
            float(np.bincount(y, minlength=len(CANDIDATES)).max() / len(y)))


def cv_acc(H, idx, rows, form, cell):
    y = np.array([CANDIDATES.index(c) for c in labels(rows, cell)])
    X = H[np.array(idx[(form, "absent", cell)])]
    g = np.array([r["sample_id"] for r in rows])
    pred = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=500))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    return float(accuracy_score(y, pred))


def main(path, rows_path, ref="std"):
    H, idx, _ = load(path)
    rows = [r for r in (json.loads(line) for line in open(rows_path))
            if r.get("probs")]
    forms = sorted({k[0] for k in idx})
    print(f"\n## {path.split('/')[-1]}  n={len(rows)}")
    print("| form | acc_b CV | a->b raw | a->b bal | b->a raw | b->a bal | "
          "baseline_b | baseline_a | std_a -> form_b (raw/bal) |")
    print("|---|---|---|---|---|---|---|---|---|")
    pipe_ref = fit(H, idx, rows, ref, "a")
    for form in forms:
        pipe = fit(H, idx, rows, form, "a")
        pipe_b = fit(H, idx, rows, form, "b")
        ab = acc(pipe, H, idx, rows, form, "b")
        ba = acc(pipe_b, H, idx, rows, form, "a")
        rf = acc(pipe_ref, H, idx, rows, form, "b")
        print(f"| {form} | {cv_acc(H, idx, rows, form, 'b'):.3f} "
              f"| {ab[0]:.3f} | {ab[1]:.3f} "
              f"| {ba[0]:.3f} | {ba[1]:.3f} "
              f"| {ab[2]:.3f} | {ba[2]:.3f} "
              f"| {rf[0]:.3f}/{rf[1]:.3f} |",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "std")
