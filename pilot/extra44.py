"""Controls the v2 review asked for, in one pass.

1. Projection ablation matched in removed energy. Removing w removes 5.4x more
   energy than removing a random direction, so the unmatched comparison cannot
   support a necessity claim. Here every removal is rescaled to a common target
   energy, in both directions: w scaled down to a random direction's energy, and
   random/class-mean/PC1 scaled up to w's energy.
2. Class-mean and PC1 directions as competitors: if any direction that is
   aligned with state structure does the same, the effect is not specific.
3. Dosing by each prompt's own final-state norm rather than a fixed median, so
   the dose-response is not an artifact of absolute injection size.
"""

import argparse
import json
import time

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from .infer import DATA, ROOT, Scorer, build_user
from .steer import fit_direction
from .steer_conflict import find_lm_head, prompts_for

KEYS = ("sufficient|wrong", "absent|a", "absent|b")


def nohint_states(run_path):
    from .probe import load
    recs, names, H = load(run_path)
    Xs, Xa = [], []
    for r in recs:
        kp = f"{r['sample_id']}|sufficient|none"
        ka = f"{r['sample_id']}|absent|none"
        if kp in H and ka in H:
            Xs.append(H[kp].astype(np.float32))
            Xa.append(H[ka].astype(np.float32))
    return np.stack(Xs), np.stack(Xa)


def extra_directions(run_path, w):
    Xs, Xa = nohint_states(run_path)
    cm = Xs.mean(0) - Xa.mean(0)
    cm /= np.linalg.norm(cm)
    X = np.concatenate([Xs, Xa])
    p = PCA(n_components=1, random_state=0).fit(X - X.mean(0))
    pc1 = p.components_[0]
    if pc1 @ cm < 0:
        pc1 = -pc1
    pc1 = pc1 / np.linalg.norm(pc1)
    return {"cm": cm.astype(np.float32), "pc1": pc1.astype(np.float32)}, X


def energies(dirs, X):
    return {k: float(np.mean(np.abs(X @ v))) for k, v in dirs.items()}


def random_orth(w, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(len(w))
    v -= (v @ w) * w
    return (v / np.linalg.norm(v)).astype(np.float32)


class SteerK:
    def __init__(self, model, head=None):
        import torch
        self.torch = torch
        self.w = None
        self.k = 1.0
        self.alpha = 0.0
        self.norm = None
        self.mode = "none"
        self.handle = (head or model.lm_head).register_forward_pre_hook(self.hook)

    def hook(self, module, args):
        if self.mode == "none":
            return None
        h = args[0]
        if self.mode == "capture":
            self.norm = float(h[:, -1, :].float().norm(dim=-1).mean().cpu())
            return None
        v = self.torch.as_tensor(self.w, dtype=h.dtype, device=h.device)
        if self.mode == "add":
            h2 = h + self.alpha * v
        elif self.mode == "add_norm":
            h2 = h + self.alpha * self.norm * v
        else:
            h2 = h - self.k * (h * v).sum(-1, keepdim=True) * v
        return (h2,) + args[1:]

    def off(self):
        self.mode = "none"

    def close(self):
        self.handle.remove()


def build_runner(args):
    if args.family == "qwen":
        scorer = Scorer(args.model)
        cand = scorer.cand_names

        def score(img, text):
            return scorer.score([img], text)[0]
    else:
        from .infer_gen import Gen
        scorer = Gen(args.model, resize=args.resize)
        cand = list(scorer.ids)

        def score(img, text):
            return scorer.score(img, text)
    return scorer, cand, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "extra44.jsonl"))
    ap.add_argument("--stats", default=str(ROOT / "results" / "extra44_stats.json"))
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[: args.limit]

    w, scale = fit_direction(args.run)
    dirs = {"w": w, "r0": random_orth(w, args.seed),
            "r1": random_orth(w, args.seed + 1)}
    extra, X = extra_directions(args.run, w)
    dirs.update(extra)
    E = energies(dirs, X)

    # removal scale per direction: k=1 is the plain projection. mw removes as
    # much energy as w does; mr0 removes as much as r0 does.
    abl = [("abl_w", "w", 1.0),
           ("abl_w_mr0", "w", E["r0"] / E["w"]),
           ("abl_r0", "r0", 1.0),
           ("abl_r0_mw", "r0", E["w"] / E["r0"])]
    if not args.lite:
        abl += [("abl_r1", "r1", 1.0),
                ("abl_r1_mw", "r1", E["w"] / E["r1"])]
    abl += [("abl_cm", "cm", 1.0),
            ("abl_cm_mw", "cm", E["w"] / E["cm"]),
            ("abl_pc1", "pc1", 1.0),
            ("abl_pc1_mw", "pc1", E["w"] / E["pc1"])]
    adds = [("add_w+0.25", "w", 0.25)]
    if not args.lite:
        adds += [("add_cm+0.25", "cm", 0.25), ("add_pc1+0.25", "pc1", 0.25)]

    scorer, cand, score = build_runner(args)
    st = SteerK(scorer.model, head=find_lm_head(scorer.model))
    print("energies", {k: round(v, 5) for k, v in E.items()},
          "scale", round(scale, 1), flush=True)

    stats = {"model": args.model, "family": args.family, "resize": args.resize,
             "energies": E, "energy_frac": {k: v / scale for k, v in E.items()},
             "removal_k": {name: k for name, _, k in abl}, "median_norm": scale}
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "family": args.family,
                                     "resize": args.resize,
                                     "candidates": ",".join(cand), "scale": scale,
                                     "energies": E,
                                     "removal_k": stats["removal_k"]}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            prompts = prompts_for(row)
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}, "norms": {}}
            for key in KEYS:
                rec["probs"][f"{key}|base"] = score(img, prompts[key])
            for name, dkey, k in abl:
                st.mode, st.w, st.k = "ablate", dirs[dkey], k
                for key in KEYS:
                    rec["probs"][f"{key}|{name}"] = score(img, prompts[key])
                st.off()
            for name, dkey, a in adds:
                st.mode, st.w, st.alpha = "add", dirs[dkey], a * scale
                for key in KEYS:
                    rec["probs"][f"{key}|{name}"] = score(img, prompts[key])
                st.off()
            # per-prompt dose: capture the conflict prompt's own final norm
            for key in ("absent|a",):
                st.mode = "capture"
                score(img, prompts[key])
                st.off()
                rec["norms"][key] = st.norm
                for a in (-0.5, -0.25, 0.25, 0.5):
                    st.mode, st.w, st.alpha = "add_norm", dirs["w"], a
                    rec["probs"][f"{key}|add_norm{a:+.2f}"] = score(img, prompts[key])
                    st.off()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    st.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
