"""Causal controls for steering along the evidence direction.

Three checks, all under the conflicting-hint pass:

1. Random directions of the same norm, orthogonalised to w. If adding w is
   special, the w effect should sit outside the random-direction spread.
2. The same direction at head-level and intermediate-layer last-token sites.
   A strong head effect with no layer effect is a readout-level result.
3. Dose-matched layer steering: at each layer, alpha is a fixed fraction of
   that layer's median last-token norm, not of the final-layer norm.
"""

import argparse
import json
import time

import numpy as np
from PIL import Image

from .constants import CANDIDATES, DISTRACTOR
from .infer import DATA, ROOT, Scorer, build_user
from .steer import Steer, fit_direction, set_mode
from .steer_conflict import find_lm_head, prompts_for

KEYS = ("sufficient|wrong", "absent|a", "absent|b")


class LayerSteer:
    def __init__(self, layer):
        self.mode = "none"
        self.vec = None
        self.alpha = 0.0
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if self.mode == "none":
            return None
        hs = output if hasattr(output, "dim") else output[0]
        h2 = hs.clone()
        h2[:, -1, :] = h2[:, -1, :] + self.alpha * self.vec.to(hs.dtype)
        return h2 if hasattr(output, "dim") else (h2,) + output[1:]

    def off(self):
        self.mode = "none"


class NormTap:
    def __init__(self, layer):
        self.mode = False
        self.norm = None
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if not self.mode:
            return None
        hs = output if hasattr(output, "dim") else output[0]
        self.norm = float(hs[:, -1, :].float().norm(dim=-1).mean().cpu())
        return None


def orthogonal_directions(w, k, seed=0):
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(k):
        v = rng.standard_normal(len(w))
        v -= (v @ w) * w
        for b in basis:
            v -= (v @ b) * b
        v /= np.linalg.norm(v)
        basis.append(v)
    return [v.astype(np.float32) for v in basis]


def build_runner(args):
    if args.family == "qwen":
        scorer = Scorer(args.model)
        lm = scorer.model.model.language_model

        def score(img, text):
            return scorer.score([img], text)[0]
    elif args.family == "phi":
        from .patch_full import find_layers
        from .phi_infer import Gen
        g = Gen(args.model, resize=args.resize)
        _, lm = find_layers(g.model)
        scorer = g

        def score(img, text):
            return g.score(img, text)
    else:
        from .infer_gen import Gen
        from .patch4 import find_layers
        g = Gen(args.model, resize=args.resize)
        _, lm = find_layers(g.model)
        scorer = g

        def score(img, text):
            return g.score(img, text)
    return scorer, lm.layers, score


def estimate_layer_scales(rows, layers, ids, score, n=16):
    """Median last-token norm per layer on absent/no-hint prompts."""
    taps = {L: NormTap(layers[L]) for L in ids}
    norms = {L: [] for L in ids}
    for row in rows[:n]:
        img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
        text = build_user(row, "none", row["absent_category"])
        for tap in taps.values():
            tap.mode, tap.norm = True, None
        score(img, text)
        for L, tap in taps.items():
            tap.mode = False
            norms[L].append(tap.norm)
    return {L: float(np.median(v)) for L, v in norms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen", "phi"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "control_steer.jsonl"))
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--randoms", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-n", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.skip:]

    w, scale = fit_direction(args.run)
    print(f"direction fitted, median final norm {scale:.1f}", flush=True)

    scorer, layers, score = build_runner(args)
    layer_ids = [int(x) for x in args.layers.split(",")]
    layer_scales = estimate_layer_scales(rows, layers, layer_ids, score, args.scale_n)
    layer_steers = {L: LayerSteer(layers[L]) for L in layer_ids}
    head_steer = Steer(scorer.model, w, scale, head=find_lm_head(scorer.model))
    print("layer scales", layer_scales, flush=True)

    randoms = orthogonal_directions(w, args.randoms, args.seed)
    # ablate must carry w: head_steer.w is rebound by later settings, so None
    # would leave the previous row's last random direction in place.
    head_settings = [("ablate", "ablate", w, 0.0)]
    head_settings += [(f"rabl{i}", "ablate", r, 0.0)
                      for i, r in enumerate(randoms)]
    head_settings += [(f"w{a:+.2f}", "add", w, a)
                      for a in (-0.5, -args.alpha, args.alpha, 0.5)]
    for i, r in enumerate(randoms):
        for a in (-args.alpha, args.alpha):
            head_settings.append((f"r{i}{a:+.2f}", "add", r, a))
    layer_settings = [(f"L{L}{s * args.alpha:+.2f}", L, s * args.alpha)
                      for L in layer_ids for s in (-1, 1)]

    t0 = time.time()
    with open(args.out, "a" if args.append else "w") as f:
        if not args.append:
            f.write(json.dumps({"meta": {
                "model": args.model, "family": args.family, "resize": args.resize,
                "candidates": ",".join(getattr(scorer, "cand_names", None)
                                       or (CANDIDATES if args.family == "phi"
                                           else list(scorer.ids))),
                "scale": scale, "alpha": args.alpha, "seed": args.seed,
                "layers": layer_ids, "layer_scales": layer_scales,
                "randoms": args.randoms}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            prompts = prompts_for(row)
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for key in KEYS:
                rec["probs"][f"{key}|base"] = score(img, prompts[key])
            for name, mode, vec, alpha in head_settings:
                if vec is not None:
                    head_steer.w = head_steer.torch.tensor(
                        vec, dtype=head_steer.torch.float32, device="cuda")
                set_mode(head_steer, mode, alpha * scale)
                for key in KEYS:
                    rec["probs"][f"{key}|{name}"] = score(img, prompts[key])
            set_mode(head_steer, "none")
            for name, L, frac in layer_settings:
                ls = layer_steers[L]
                ls.vec = head_steer.torch.tensor(
                    w, dtype=head_steer.torch.float32, device="cuda")
                ls.alpha = frac * layer_scales[L]
                ls.mode = "add"
                for key in KEYS:
                    rec["probs"][f"{key}|{name}"] = score(img, prompts[key])
                ls.off()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    head_steer.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
