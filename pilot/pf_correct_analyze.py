"""Does the axis move only when the hint conflicts?  (correct-hint cell)

Merges the six-form battery with the extra sufficient|correct cell and
compares three within-form axes on the same scenes:

  v0 = mean(sufficient none)  - mean(absent none)
  vc = mean(sufficient correct) - mean(absent a)     hint agrees with image
  va = mean(sufficient wrong)   - mean(absent a)     hint conflicts

If the rotation is caused by conflict rather than by adding a hint sentence,
vc stays close to v0 while va rotates.  Behaviour: accuracy in the correct
cell against the no-hint ceiling and the wrong-hint drop.
"""

import json
import sys
from collections import defaultdict

import numpy as np

from .constants import CANDIDATES


def load_npz(path, store):
    z = np.load(path, allow_pickle=True)
    for key, h in zip((str(k) for k in z["keys"]), z["H"]):
        store[key] = h.astype(np.float32)


def index(store):
    idx, scenes = defaultdict(list), defaultdict(list)
    for i, key in enumerate(store):
        s, form, ev, hint = key.split("|")
        idx[(form, ev, hint)].append(i)
        scenes[(form, ev, hint)].append(s)
    return idx, scenes


def axes(H, idx, form, sel):
    def m(ev, hint):
        rows = np.array(idx[(form, ev, hint)])
        return H[rows if sel is None else rows[sel]].mean(0)
    v0 = m("sufficient", "none") - m("absent", "none")
    vc = m("sufficient", "correct") - m("absent", "a")
    va = m("sufficient", "wrong") - m("absent", "a")
    return v0, vc, va


def unit(v):
    return v / np.linalg.norm(v)


def stats(H, idx, form, sel=None):
    v0, vc, va = axes(H, idx, form, sel)
    u0, uc, ua = unit(v0), unit(vc), unit(va)
    return {
        "len_c": float(np.linalg.norm(vc) / np.linalg.norm(v0)),
        "len_a": float(np.linalg.norm(va) / np.linalg.norm(v0)),
        "ang_c": float(np.degrees(np.arccos(np.clip(uc @ u0, -1, 1)))),
        "ang_a": float(np.degrees(np.arccos(np.clip(ua @ u0, -1, 1)))),
        "cos_c_a": float(uc @ ua),
    }


def behaviour(recs, form):
    cand = CANDIDATES
    out = {}
    for cell, key in (("none", f"{form}|sufficient|none"),
                      ("correct", f"{form}|sufficient|correct"),
                      ("wrong", f"{form}|sufficient|wrong")):
        p = np.array([r["probs"][key] for r in recs])
        pred = p.argmax(1)
        gold = np.array([cand.index(r["gold"]) for r in recs])
        out[f"acc_{cell}"] = float((pred == gold).mean())
        out[f"refuse_{cell}"] = float((pred == cand.index("unknown")).mean())
    out["gain_correct"] = out["acc_correct"] - out["acc_none"]
    out["loss_wrong"] = out["acc_wrong"] - out["acc_none"]
    return out


def main(beh_jsonl, extra_jsonl, npz_main, npz_extra):
    recs = [r for r in (json.loads(line) for line in open(beh_jsonl))
            if r.get("probs")]
    extra = [r for r in (json.loads(line) for line in open(extra_jsonl))
             if r.get("probs")]
    assert len(recs) == len(extra)
    for r, e in zip(recs, extra):
        assert r["sample_id"] == e["sample_id"]
        r["probs"].update(e["probs"])
    store = {}
    load_npz(npz_main, store)
    load_npz(npz_extra, store)
    H = np.stack(list(store.values()))
    idx, scenes = index(store)
    forms = sorted({k[0] for k in idx})
    name = beh_jsonl.split("/")[-1]
    print(f"\n## {name}: correct-hint cell")
    print("| form | acc_none | acc_correct | gain | acc_wrong | loss | refuse_correct |")
    print("|---|---|---|---|---|---|---|")
    beh = {}
    for form in forms:
        b = behaviour(recs, form)
        beh[form] = b
        print(f"| {form} | {b['acc_none']:.3f} | {b['acc_correct']:.3f} "
              f"| {b['gain_correct']:+.3f} | {b['acc_wrong']:.3f} "
              f"| {b['loss_wrong']:+.3f} | {b['refuse_correct']:.3f} |")
    print("\n| form | len_c | len_a | ang_c | ang_a | cos(vc,va) |")
    print("|---|---|---|---|---|---|")
    n = len(idx[(forms[0], "absent", "none")])
    rng = np.random.default_rng(0)
    geo = {}
    for form in forms:
        s = stats(H, idx, form)
        boots = [stats(H, idx, form, rng.integers(0, n, n)) for _ in range(400)]
        ci = {k: np.percentile([b[k] for b in boots], [2.5, 97.5])
              for k in s}
        geo[form] = {"point": s, "ci": {k: v.tolist() for k, v in ci.items()}}
        print(f"| {form} | {s['len_c']:.2f} "
              f"[{ci['len_c'][0]:.2f},{ci['len_c'][1]:.2f}] "
              f"| {s['len_a']:.2f} [{ci['len_a'][0]:.2f},{ci['len_a'][1]:.2f}] "
              f"| {s['ang_c']:.0f} [{ci['ang_c'][0]:.0f},{ci['ang_c'][1]:.0f}] "
              f"| {s['ang_a']:.0f} [{ci['ang_a'][0]:.0f},{ci['ang_a'][1]:.0f}] "
              f"| {s['cos_c_a']:.2f} [{ci['cos_c_a'][0]:.2f},{ci['cos_c_a'][1]:.2f}] |")
    out = beh_jsonl.replace(".jsonl", "_correct_analysis.json")
    json.dump({"behaviour": beh, "geometry": geo}, open(out, "w"), indent=1)
    print("\nsaved", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
