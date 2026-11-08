"""Scene-bootstrap CIs for prompt-form behaviour and evidence-axis geometry.

Reads a run from prompt_forms.py. Resamples scenes with replacement and
reports, per form, adoption/refusal rates, the mean-difference length ratio
and rotation angle, and the paired contrast against the std form.
"""

import sys

import numpy as np

from .constants import CANDIDATES, DISTRACTOR, REFUSAL
from .prompt_forms_analyze import load_jsonl, load_states

UNK = CANDIDATES.index(REFUSAL)


def scene_stats(rec, form):
    p = rec["probs"]
    return {
        "adopt_a": int(np.argmax(p[f"{form}|absent|a"]) ==
                       CANDIDATES.index(rec["absent_hint"])),
        "adopt_b": int(np.argmax(p[f"{form}|absent|b"]) ==
                       CANDIDATES.index(DISTRACTOR[rec["absent_hint"]])),
        "refuse_none": int(np.argmax(p[f"{form}|absent|none"]) == UNK),
        "gold_wrong": int(np.argmax(p[f"{form}|sufficient|wrong"]) ==
                          CANDIDATES.index(rec["gold"])),
        "adopt_wrong": int(np.argmax(p[f"{form}|sufficient|wrong"]) ==
                           CANDIDATES.index(DISTRACTOR[rec["gold"]])),
    }


def boot_ci(x, n_boot, rng):
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean()
                   for _ in range(n_boot)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def main(jsonl, n_boot=2000, seed=0):
    recs = load_jsonl(jsonl)
    H, idx, scenes, forms = load_states(jsonl.replace(".jsonl", ".npz"))
    rng = np.random.default_rng(seed)
    n = len(recs)

    beh = {f: [scene_stats(r, f) for r in recs] for f in forms}
    raw = {}
    for form in forms:
        raw[form] = [H[idx[(form, ev, hint)]] for ev, hint in
                     (("sufficient", "none"), ("absent", "none"),
                      ("sufficient", "wrong"), ("absent", "a"))]

    def length_ratio(form, sel=None):
        m1, m0, mw, ma = raw[form]
        if sel is None:
            u0 = m1.mean(0) - m0.mean(0)
            ua = mw.mean(0) - ma.mean(0)
        else:
            u0 = m1[sel].mean(0) - m0[sel].mean(0)
            ua = mw[sel].mean(0) - ma[sel].mean(0)
        return np.linalg.norm(ua) / np.linalg.norm(u0), u0, ua

    keys = ["adopt_a", "adopt_b", "adopt_wrong", "refuse_none", "gold_wrong"]
    print(f"\n## {jsonl}: behaviour, mean [95% CI] (n={n})\n")
    print("| form | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for form in forms:
        cells = []
        for k in keys:
            x = np.array([s[k] for s in beh[form]])
            m, lo, hi = boot_ci(x, n_boot, rng)
            cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"| {form} | " + " | ".join(cells) + " |")

    print(f"\n## {jsonl}: geometry, mean [95% CI]\n")
    print("| form | length_ratio | angle_deg |")
    print("|---|---|---|")
    for form in forms:
        lr, u0, ua = length_ratio(form)
        ang = np.degrees(np.arccos(np.clip(
            ua @ u0 / (np.linalg.norm(ua) * np.linalg.norm(u0)), -1, 1)))
        lrs, angs = [], []
        for _ in range(n_boot):
            sel = rng.integers(0, n, n)
            l, a, b = length_ratio(form, sel)
            lrs.append(l)
            angs.append(np.degrees(np.arccos(np.clip(
                b @ a / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        print(f"| {form} | {lr:.3f} [{np.percentile(lrs,2.5):.3f},"
              f"{np.percentile(lrs,97.5):.3f}] | {ang:.1f} "
              f"[{np.percentile(angs,2.5):.1f},{np.percentile(angs,97.5):.1f}] |")

    print(f"\n## {jsonl}: paired contrast vs std, mean [95% CI]\n")
    print("| form | d_adopt_b | d_length_ratio |")
    print("|---|---|---|")
    std_b = np.array([s["adopt_b"] for s in beh["std"]])
    std_lr = length_ratio("std")[0]
    for form in forms:
        if form == "std":
            continue
        xb = np.array([s["adopt_b"] for s in beh[form]])
        d, dl = [], []
        for _ in range(n_boot):
            sel = rng.integers(0, n, n)
            d.append(std_b[sel].mean() - xb[sel].mean())
            dl.append(length_ratio("std", sel)[0] - length_ratio(form, sel)[0])
        d, dl = np.array(d), np.array(dl)
        print(f"| {form} | {std_b.mean() - xb.mean():+.3f} "
              f"[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] | "
              f"{std_lr - length_ratio(form)[0]:+.3f} "
              f"[{np.percentile(dl,2.5):+.3f},{np.percentile(dl,97.5):+.3f}] |")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
