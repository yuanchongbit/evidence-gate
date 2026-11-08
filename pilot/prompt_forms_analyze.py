"""Behaviour and evidence-axis geometry across prompt forms.

Reads a run written by prompt_forms.py (or prompt_variants.py) plus its npz of
final states.  For each form reports refusal, hint adoption and gold accuracy
in the five cells, and the geometry of the evidence axis: conflict/clean length
ratio, rotation angle, cosine between the two conflicting axes, within-form
frozen probe transfer, and cross-form transfer of the standard-form probe.
"""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CANDIDATES, DISTRACTOR, REFUSAL


def load_jsonl(path):
    recs = []
    for line in open(path):
        rec = json.loads(line)
        if "probs" in rec:
            recs.append(rec)
    return recs


def load_states(path):
    z = np.load(path, allow_pickle=True)
    idx, scenes = defaultdict(list), defaultdict(list)
    forms = set()
    for i, key in enumerate(str(k) for k in z["keys"]):
        parts = key.split("|")
        if len(parts) == 4:
            s, form, ev, hint = parts
        elif len(parts) == 3:
            s, ev, hint = parts
            form = "main"
        else:
            raise ValueError(key)
        idx[(form, ev, hint)].append(i)
        scenes[(form, ev, hint)].append(s)
        forms.add(form)
    return z["H"].astype(np.float32), idx, scenes, sorted(forms)


def cell(rec, form, ev, hint):
    key = f"{form}|{ev}|{hint}" if form != "main" else f"{ev}|{hint}"
    return rec["probs"][key]


def stated(rec, ev, hint):
    if ev == "absent" and hint == "a":
        return rec["absent_hint"]
    if ev == "absent" and hint == "b":
        return DISTRACTOR[rec["absent_hint"]]
    if ev == "sufficient" and hint == "wrong":
        return DISTRACTOR[rec["gold"]]
    return None


def behaviour(recs, form):
    cand = CANDIDATES
    unk = cand.index(REFUSAL)
    acc = defaultdict(list)
    for rec in recs:
        for ev, hint in (("absent", "none"), ("absent", "a"), ("absent", "b"),
                         ("sufficient", "none"), ("sufficient", "wrong")):
            p = cell(rec, form, ev, hint)
            pred = int(np.argmax(p))
            acc[f"{ev}_{hint}"].append(pred)
        for ev, hint in (("absent", "a"), ("absent", "b"),
                         ("sufficient", "wrong")):
            p = cell(rec, form, ev, hint)
            acc[f"adopt_{ev}_{hint}"].append(
                int(np.argmax(p)) == cand.index(stated(rec, ev, hint)))
    out = {}
    for k, preds in acc.items():
        preds = np.array(preds)
        if k.startswith("adopt"):
            out[k] = float(preds.mean())
        else:
            out[f"refuse_{k}"] = float((preds == unk).mean())
            if k.startswith("sufficient"):
                gold = np.array([r["gold"] for r in recs])
                out[f"gold_{k}"] = float((preds == np.array(
                    [cand.index(g) for g in gold])).mean())
    out["adopt_absent"] = 0.5 * (out["adopt_absent_a"] + out["adopt_absent_b"])
    return out


def fit_probe(H, idx, form):
    X0 = H[idx[(form, "absent", "none")]]
    X1 = H[idx[(form, "sufficient", "none")]]
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    pipe.fit(X, y)
    return pipe


def fit_direction(H, idx, form):
    X0 = H[idx[(form, "absent", "none")]]
    X1 = H[idx[(form, "sufficient", "none")]]
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=3000))
    pipe.fit(X, y)
    scaler, clf = pipe[0], pipe[-1]
    w = clf.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)


def conflict_states(H, idx, form, hint="a"):
    return np.vstack([H[idx[(form, "sufficient", "wrong")]],
                      H[idx[(form, "absent", hint)]]]), np.r_[
        np.ones(len(idx[(form, "sufficient", "wrong")])),
        np.zeros(len(idx[(form, "absent", hint)]))]


def geometry(H, idx, forms, ref="std"):
    out = {}
    ref_pipe = fit_probe(H, idx, ref) if ref in forms else None
    ref_dir = fit_direction(H, idx, ref) if ref in forms else None
    for form in forms:
        def mat(ev, hint):
            return H[idx[(form, ev, hint)]]

        v0 = mat("sufficient", "none").mean(0) - mat("absent", "none").mean(0)
        va = mat("sufficient", "wrong").mean(0) - mat("absent", "a").mean(0)
        vb = mat("sufficient", "wrong").mean(0) - mat("absent", "b").mean(0)
        n0, na, nb = np.linalg.norm(v0), np.linalg.norm(va), np.linalg.norm(vb)
        pipe = fit_probe(H, idx, form)
        Xc, yc = conflict_states(H, idx, form, "a")
        d = {
            "length_ratio": float(na / n0),
            "angle_deg": float(np.degrees(np.arccos(
                np.clip(va @ v0 / (na * n0), -1, 1)))),
            "cos_hint_pairs": float(va @ vb / (na * nb)),
            "frozen_probe_conflict": float(
                roc_auc_score(yc, pipe.decision_function(Xc))),
        }
        if ref is not None:
            d["cos_to_ref_dir"] = float(
                fit_direction(H, idx, form) @ ref_dir)
            d["ref_probe_transfer"] = float(
                roc_auc_score(yc, ref_pipe.decision_function(Xc)))
        out[form] = d
    return out


def print_tables(name, beh, geo):
    print(f"\n## {name}: behaviour\n")
    cols = ["refuse_absent_none", "refuse_absent_a", "refuse_absent_b",
            "adopt_absent_a", "adopt_absent_b",
            "gold_sufficient_none", "gold_sufficient_wrong",
            "refuse_sufficient_none", "refuse_sufficient_wrong",
            "adopt_sufficient_wrong"]
    print("| form | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for form in beh:
        print(f"| {form} | " + " | ".join(
            f"{beh[form][c]:.3f}" for c in cols) + " |")
    print(f"\n## {name}: geometry\n")
    cols = ["length_ratio", "angle_deg", "cos_hint_pairs",
            "frozen_probe_conflict", "cos_to_ref_dir", "ref_probe_transfer"]
    print("| form | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for form in geo:
        print(f"| {form} | " + " | ".join(
            (f"{geo[form][c]:.3f}" if c in geo[form] else "-") for c in cols) + " |")


def main(jsonl, ref="std"):
    stem = jsonl.rsplit(".jsonl", 1)[0]
    recs = load_jsonl(jsonl)
    H, idx, scenes, forms = load_states(stem + ".npz")
    if ref not in forms:
        ref = forms[0]
    beh = {f: behaviour(recs, f) for f in forms}
    geo = geometry(H, idx, forms, ref)
    name = stem.split("/")[-1]
    print_tables(name, beh, geo)
    json.dump({"behaviour": beh, "geometry": geo, "n": len(recs)},
              open(stem + "_analysis.json", "w"), indent=1)
    print("\nsaved", stem + "_analysis.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "std")
