"""Split-half causal check: fit the direction on half the scenes, then steer
only the held-out half. If the dose-response survives, the effect is not a
property of the samples the direction was fitted on."""

import argparse
import json
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .infer import DATA, ROOT, Scorer, build_user
from .probe import load
from .steer import Steer, set_mode


def fit_direction(run_path, keep, C=0.01):
    recs, _, H = load(run_path)
    X, y = [], []
    for r in recs:
        if r["sample_id"] not in keep:
            continue
        for key, lab in (("sufficient|none", 1), ("absent|none", 0)):
            X.append(H[f"{r['sample_id']}|{key}"].astype(np.float32))
            y.append(lab)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    pipe.fit(X, y)
    w = pipe[-1].coef_[0] / np.maximum(pipe[0].scale_, 1e-6)
    w = w / np.linalg.norm(w)
    return w.astype(np.float32), float(np.median([np.linalg.norm(x) for x in X]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "steer_half.jsonl"))
    ap.add_argument("--alphas", default="-0.5,-0.25,0.25,0.5")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    recs, _, _ = load(args.run)
    ids = sorted({r["sample_id"] for r in recs})
    perm = np.random.default_rng(args.split_seed).permutation(len(ids))
    keep = {ids[i] for i in perm[: len(ids) // 2]}
    cands = [r for r in rows if r.get("absent_category")
             and r["sample_id"] not in keep]
    w, scale = fit_direction(args.run, keep)
    print(f"fit on {len(keep)} ids, steer {len(cands)} held-out, norm {scale:.1f}",
          flush=True)

    scorer = Scorer(args.model)
    st = Steer(scorer.model, w, scale)
    settings = [("base", "none", 0.0), ("ablate", "ablate", 0.0)]
    settings += [(f"a{a:+.2f}", "add", a) for a in [float(x) for x in args.alphas.split(",")]]

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(scorer.cand_names),
                                     "scale": scale, "alphas": args.alphas,
                                     "split_seed": args.split_seed,
                                     "fit_ids": len(keep)}}) + "\n")
        for n, row in enumerate(cands, 1):
            orig = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            acat = row["absent_category"]
            prompts = {"sufficient|none": build_user(row, "none"),
                       "absent|none": build_user(row, "none", acat)}
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": acat, "probs": {}}
            for name, mode, alpha in settings:
                set_mode(st, mode, alpha * scale)
                for key, text in prompts.items():
                    rec["probs"][f"{key}|{name}"] = scorer.score([orig], text)[0]
            set_mode(st, "none")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(cands)} {time.time() - t0:.0f}s", flush=True)
    st.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
