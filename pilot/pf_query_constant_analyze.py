"""Geometry and behaviour for the constant-colour queried-category control.

Adds vq (queried-category sentence with one colour fixed across arms) to the
four axes of pf_query_control_analyze:

  v0  no-hint evidence axis (main battery)
  vn  neutral extra sentence
  vo  assertion about another object, constant colour
  vq  assertion about the queried category, constant colour
  va  nl conflict axis (colour changes between arms)

The contrast vq - vo isolates the object the sentence is about; vq vs va
isolates the change of asserted colour between arms.
"""

import argparse
import json

import numpy as np

from .constants import CANDIDATES, REFUSAL
from .pf_query_control_analyze import angle, cell, load_npz, unit

UNK = CANDIDATES.index(REFUSAL)


def boot_stats(V0, VN, VO, VQ, VA, n_boot, rng):
    n = len(V0)
    stats = {k: [] for k in (
        "len_n", "len_o", "len_q",
        "ang_n", "ang_o", "ang_q",
        "d_ang_qn", "d_len_qn", "d_ang_qo", "d_len_qo",
        "cos_no", "cos_n_q", "cos_o_q", "cos_n_a", "cos_o_a", "cos_q_a")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v0, vn, vo, vq, va = (V[idx].mean(0) for V in (V0, VN, VO, VQ, VA))
        l0 = np.linalg.norm(v0)
        stats["len_n"].append(np.linalg.norm(vn) / l0)
        stats["len_o"].append(np.linalg.norm(vo) / l0)
        stats["len_q"].append(np.linalg.norm(vq) / l0)
        an, ao, aq = angle(vn, v0), angle(vo, v0), angle(vq, v0)
        stats["ang_n"].append(an)
        stats["ang_o"].append(ao)
        stats["ang_q"].append(aq)
        stats["d_ang_qn"].append(aq - an)
        stats["d_len_qn"].append(np.linalg.norm(vq) / l0 - np.linalg.norm(vn) / l0)
        stats["d_ang_qo"].append(aq - ao)
        stats["d_len_qo"].append(np.linalg.norm(vq) / l0 - np.linalg.norm(vo) / l0)
        stats["cos_no"].append(unit(vn) @ unit(vo))
        stats["cos_n_q"].append(unit(vn) @ unit(vq))
        stats["cos_o_q"].append(unit(vo) @ unit(vq))
        stats["cos_n_a"].append(unit(vn) @ unit(va))
        stats["cos_o_a"].append(unit(vo) @ unit(va))
        stats["cos_q_a"].append(unit(vq) @ unit(va))
    return {k: np.array(v) for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-jsonl", required=True)
    ap.add_argument("--main-npz", required=True)
    ap.add_argument("--ctrl-jsonl", required=True)
    ap.add_argument("--ctrl-npz", required=True)
    ap.add_argument("--query-jsonl", required=True)
    ap.add_argument("--query-npz", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    main_store = load_npz(args.main_npz)
    ctrl_store = load_npz(args.ctrl_npz)
    q_store = load_npz(args.query_npz)
    main_recs = {r["sample_id"]: r
                 for r in (json.loads(line) for line in open(args.main_jsonl))
                 if r.get("probs")}
    ctrl_recs = [r for r in (json.loads(line) for line in open(args.ctrl_jsonl))
                 if r.get("probs")]
    q_recs = {r["sample_id"]: r
              for r in (json.loads(line) for line in open(args.query_jsonl))
              if r.get("probs")}
    recs = [r for r in ctrl_recs
            if r["sample_id"] in q_recs
            and f"{r['sample_id']}|neutral|absent|none" in ctrl_store
            and f"{r['sample_id']}|query|absent|none" in q_store]
    print(f"\n## n={len(recs)}")

    V0, VN, VO, VQ, VA = [], [], [], [], []
    for r in recs:
        sid = r["sample_id"]
        V0.append(cell(main_store, sid, "std", "sufficient", "none") -
                  cell(main_store, sid, "std", "absent", "none"))
        VN.append(cell(ctrl_store, sid, "neutral", "sufficient", "none") -
                  cell(ctrl_store, sid, "neutral", "absent", "none"))
        VO.append(cell(ctrl_store, sid, "other", "sufficient", "none") -
                  cell(ctrl_store, sid, "other", "absent", "none"))
        VQ.append(cell(q_store, sid, "query", "sufficient", "none") -
                  cell(q_store, sid, "query", "absent", "none"))
        VA.append(cell(main_store, sid, "nl", "sufficient", "wrong") -
                  cell(main_store, sid, "nl", "absent", "a"))
    V0, VN, VO, VQ, VA = (np.stack(v).astype(np.float64)
                          for v in (V0, VN, VO, VQ, VA))

    rng = np.random.default_rng(args.seed)
    bs = boot_stats(V0, VN, VO, VQ, VA, args.n_boot, rng)

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    print("\n| axis | length ratio [95% CI] | angle to v0 (deg) [95% CI] |")
    print("|---|---|---|")
    for name, lk, ak in (("neutral extra sentence", "len_n", "ang_n"),
                         ("other-object assertion", "len_o", "ang_o"),
                         ("query assertion, constant colour", "len_q", "ang_q")):
        lo, hi = ci(bs[lk])
        alo, ahi = ci(bs[ak])
        print(f"| {name} | {bs[lk].mean():.3f} [{lo:.3f},{hi:.3f}] | "
              f"{bs[ak].mean():.1f} [{alo:.1f},{ahi:.1f}] |")
    v0, va = V0.mean(0), VA.mean(0)
    print(f"| queried conflict (nl) | "
          f"{np.linalg.norm(va)/np.linalg.norm(v0):.3f} | {angle(va, v0):.1f} |")

    print("\n| pair | cosine [95% CI] |")
    print("|---|---|")
    for name, k in (("neutral vs other", "cos_no"),
                    ("neutral vs query", "cos_n_q"),
                    ("other vs query", "cos_o_q"),
                    ("neutral vs conflict", "cos_n_a"),
                    ("other vs conflict", "cos_o_a"),
                    ("query vs conflict", "cos_q_a")):
        lo, hi = ci(bs[k])
        print(f"| {name} | {bs[k].mean():+.3f} [{lo:+.3f},{hi:+.3f}] |")

    print("\n| paired contrast | estimate [95% CI] |")
    print("|---|---|")
    for name, k in (("query - neutral, angle (deg)", "d_ang_qn"),
                    ("query - neutral, length ratio", "d_len_qn"),
                    ("query - other, angle (deg)", "d_ang_qo"),
                    ("query - other, length ratio", "d_len_qo")):
        lo, hi = ci(bs[k])
        print(f"| {name} | {bs[k].mean():+.2f} [{lo:+.2f},{hi:+.2f}] |")

    def stats(rec, key, stated=None):
        p = np.array(rec["probs"][key])
        y = CANDIDATES.index(rec["gold"])
        adopted = int(p.argmax() == CANDIDATES.index(stated)) if stated else np.nan
        return (int(p.argmax() == UNK), int(p.argmax() == y), adopted)

    print("\n| cell | refusal | gold acc | answers stated colour |")
    print("|---|---|---|---|")
    for form, store in (("neutral", "ctrl"), ("other", "ctrl"), ("query", "q")):
        for ev in ("absent", "sufficient"):
            rows_s = []
            for r in recs:
                rec = r if store == "ctrl" else q_recs[r["sample_id"]]
                stated = None if form == "neutral" else r["absent_hint"]
                rows_s.append(stats(rec, f"{form}|{ev}|none", stated))
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
    main()
