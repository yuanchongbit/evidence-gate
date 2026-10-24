"""Layer-wise patching: is the no-hint state enough before the final layers?

Replacing the state at the lm_head input is close to tautological -- the output
is a function of exactly that vector. Patching at an intermediate decoder layer
is not: the remaining layers still transform the pasted state, so a restored
refusal means the evidence was carried forward rather than recomputed.

For each sample we capture the no-hint last-position state at layer L, then
paste it into the hinted run at the same layer. A shuffled-source condition
(another sample's no-hint state at the same layer) separates "this sample's
evidence" from "any absent-evidence state".
"""

import argparse
import json
import time

from PIL import Image

from .infer import ROOT, Scorer, build_user


class Tap:
    def __init__(self, layer):
        self.mode = None
        self.vec = None
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if self.mode is None:
            return None
        hs = output if hasattr(output, "dim") else output[0]
        if self.mode == "capture":
            self.vec = hs[:, -1, :].detach().float().cpu().numpy()[0]
            return None
        h2 = hs.clone()
        h2[:, -1, :] = self.state.to(hs.dtype)
        return h2 if hasattr(output, "dim") else (h2,) + output[1:]

    def set(self, vec):
        import torch
        self.mode = "paste"
        self.state = torch.tensor(vec, dtype=torch.float32, device="cuda")

    def off(self):
        self.mode = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--layers", default="24,32")
    ap.add_argument("--out", default=str(ROOT / "runs" / "patch3.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[:args.limit]
    scorer = Scorer(args.model)
    lm = scorer.model.model.language_model
    layers = [int(x) for x in args.layers.split(",")]
    taps = {L: Tap(lm.layers[L]) for L in layers}
    print("n_layers", len(lm.layers), "probe", layers, flush=True)

    prev = None
    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "kind": "patching3",
                                     "layers": layers}}) + "\n")
        for n, row in enumerate(rows, 1):
            sid = row["sample_id"]
            acat = row.get("absent_category")
            acolor = row.get("absent_hint")
            if not acat or not acolor:
                continue
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            text_none = build_user(row, "none", acat)
            text_hint = build_user(row, "color", acat, acolor)
            rec = {"sample_id": sid, "probs": {}}
            cap = {}
            for L, tap in taps.items():
                tap.mode = "capture"
                tap.vec = None
                scorer.score([img], text_none)
                tap.off()
                cap[L] = tap.vec
            for L, tap in taps.items():
                tap.set(cap[L])
                rec["probs"][f"absent|a<-same@L{L}"] = scorer.score([img], text_hint)[0]
                tap.off()
            if prev is not None:
                for L, tap in taps.items():
                    tap.set(prev[L])
                    rec["probs"][f"absent|a<-shuf@L{L}"] = scorer.score([img], text_hint)[0]
                    tap.off()
            prev = cap
            rec["probs"]["absent|a"] = scorer.score([img], text_hint)[0]
            rec["probs"]["absent|none"] = scorer.score([img], text_none)[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
