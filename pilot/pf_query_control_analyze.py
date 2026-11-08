"""Geometry and behaviour of the query-specificity control.

Compares four axes on the same scenes:

  v0  no-hint evidence axis (main battery)
  vn  neutral extra sentence (no object, no colour, not the query)
  vo  assertion about another object ("the sofa is red"), not the query
  va  assertion about the queried category (nl conflict cells, main battery)

If vn and vo stay close to v0 while va rotates, the rotation is specific to
an assertion about the queried category rather than to extra text or to any
colour word.
"""

import json
import sys

import numpy as np

from .constants import CANDIDATES, REFUSAL

UNK = CANDIDATES.index(REFUSAL)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    H = z["H"].astype(np.float32)
    d = {str(k): H[i] for i, k in enumerate(z["keys"])}
    return d


def cell(store, sid, form, ev, hint):
    return store[f"{sid}|{form}|{ev}|{hint}"]


def unit(v):
    return v / np.linalg.norm(v)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def boot_stats(V0, VN, VO, VA, n_boot, rng):
    n = len(V0)
    stats = {k: [] for k in ("len_n", "len_o", "ang_n", "ang_o",
                             "cos_no", "cos_n_a", "cos_o_a",
                             "d_ang", "d_len")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v0, vn, vo, va = (V[idx].mean(0)
                          for V in (V0, VN, VO, VA))
        stats["len_n"].append(np.linalg.norm(vn) / np.linalg.norm(v0))
        stats["len_o"].append(np.linalg.norm(vo) / np.linalg.norm(v0))
        an, ao = angle(vn, v0), angle(vo, v0)
        stats["ang_n"].append(an)
        stats["ang_o"].append(ao)
        stats["d_ang"].append(ao - an)
        stats["d_len"].append(
            np.linalg.norm(vo) / np.linalg.norm(v0)
            - np.linalg.norm(vn) / np.linalg.norm(v0))
        stats["cos_no"].append(unit(vn) @ unit(vo))
        stats["cos_n_a"].append(unit(vn) @ unit(va))
        stats["cos_o_a"].append(unit(vo) @ unit(va))
    return {k: np.array(v) for k, v in stats.items()}


def main(main_jsonl, main_npz, ctrl_jsonl, ctrl_npz, n_boot=1000, seed=0):
    main_store = load_npz(main_npz)
    ctrl_store = load_npz(ctrl_npz)
    ctrl_recs = [r for r in (json.loads(line) for line in open(ctrl_jsonl))
                 if r.get("probs")]
    main_recs = {r["sample_id"]: r
                 for r in (json.loads(line) for line in open(main_jsonl))
                 if r.get("probs")}
    recs = [r for r in ctrl_recs
            if f"{r['sample_id']}|neutral|absent|none" in ctrl_store]
    print(f"\n## {ctrl_jsonl}  n={len(recs)}")

    V0, VN, VO, VA = [], [], [], []
    for r in recs:
        sid = r["sample_id"]
        V0.append(cell(main_store, sid, "std", "sufficient", "none") -
                  cell(main_store, sid, "std", "absent", "none"))
        VN.append(cell(ctrl_store, sid, "neutral", "sufficient", "none") -
                  cell(ctrl_store, sid, "neutral", "absent", "none"))
        VO.append(cell(ctrl_store, sid, "other", "sufficient", "none") -
                  cell(ctrl_store, sid, "other", "absent", "none"))
        VA.append(cell(main_store, sid, "nl", "sufficient", "wrong") -
                  cell(main_store, sid, "nl", "absent", "a"))
    V0, VN, VO, VA = (np.stack(v).astype(np.float64) for v in (V0, VN, VO, VA))

    rng = np.random.default_rng(seed)
    bs = boot_stats(V0, VN, VO, VA, n_boot, rng)

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    print("\n| axis | length ratio [95% CI] | angle to v0 (deg) [95% CI] |")
    print("|---|---|---|")
    for name, lk, ak in (("neutral extra sentence", "len_n", "ang_n"),
                         ("other-object assertion", "len_o", "ang_o")):
        lo, hi = ci(bs[lk])
        alo, ahi = ci(bs[ak])
        print(f"| {name} | {bs[lk].mean():.3f} [{lo:.3f},{hi:.3f}] | "
              f"{bs[ak].mean():.1f} [{alo:.1f},{ahi:.1f}] |")
    v0, va = V0.mean(0), VA.mean(0)
    print(f"| queried-category assertion (nl) | "
          f"{np.linalg.norm(va)/np.linalg.norm(v0):.3f} | {angle(va, v0):.1f} |")

    print("\n| pair | cosine [95% CI] |")
    print("|---|---|")
    for name, k in (("neutral vs other", "cos_no"),
                    ("neutral vs queried", "cos_n_a"),
                    ("other vs queried", "cos_o_a")):
        lo, hi = ci(bs[k])
        print(f"| {name} | {bs[k].mean():+.3f} [{lo:+.3f},{hi:+.3f}] |")

    print("\n| paired contrast (other - neutral) | estimate [95% CI] |")
    print("|---|---|")
    for name, k in (("angle to v0 (deg)", "d_ang"),
                    ("length ratio", "d_len")):
        lo, hi = ci(bs[k])
        print(f"| {name} | {bs[k].mean():+.2f} [{lo:+.2f},{hi:+.2f}] |")

    def stats(rec, key, stated=None):
        p = np.array(rec["probs"][key])
        y = CANDIDATES.index(rec["gold"])
        adopted = int(p.argmax() == CANDIDATES.index(stated)) if stated else np.nan
        return (int(p.argmax() == UNK), int(p.argmax() == y), adopted)

    print("\n| cell | refusal | gold acc | answers stated colour |")
    print("|---|---|---|---|")
    for form in ("neutral", "other"):
        for ev in ("absent", "sufficient"):
            rows_s = [stats(r, f"{form}|{ev}|none",
                            r["absent_hint"] if form == "other" else None)
                      for r in recs]
            s = np.array([(a, b) for a, b, _ in rows_s], dtype=float)
            adopted = [c for _, _, c in rows_s if not np.isnan(c)]
            a_txt = f"{np.mean(adopted):.3f}" if adopted else "—"
            print(f"| {form} {ev} | {s[:,0].mean():.3f} | "
                  f"{s[:,1].mean():.3f} | {a_txt} |")
    s0 = np.array([stats(main_recs[r["sample_id"]], "std|absent|none")
                   for r in recs])
    s1 = np.array([stats(main_recs[r["sample_id"]], "std|sufficient|none")
                   for r in recs])
    print(f"| no-hint absent | {s0[:,0].mean():.3f} | — | — |")
    print(f"| no-hint sufficient | — | {s1[:,1].mean():.3f} | — |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 1000)
