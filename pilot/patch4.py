"""Layer-wise patching on a non-Qwen-VL family (LLaVA-OneVision).

Same protocol as patch3.py, but the model is loaded through infer_gen.Gen
(AutoModelForImageTextToText) and the decoder stack is located by name.
Purpose: check whether the "late write" depth profile is Qwen-specific.
"""

import argparse
import json
import time

from PIL import Image

from .infer import ROOT, build_user
from .infer_gen import Gen
from .patch3 import Tap


def find_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("language_model") and hasattr(mod, "layers"):
            return name, mod
    raise RuntimeError("no decoder stack found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    ap.add_argument("--layers", default="18,24,26")
    ap.add_argument("--out", default=str(ROOT / "runs" / "patch4_llava.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resize", type=int, default=512)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    g = Gen(args.model, resize=args.resize)
    name, lm = find_layers(g.model)
    layers = [int(x) for x in args.layers.split(",")]
    print("stack", name, "n_layers", len(lm.layers), "probe", layers, flush=True)
    taps = {L: Tap(lm.layers[L]) for L in layers}

    prev = None
    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "kind": "patching4",
                                     "layers": layers, "resize": args.resize}}) + "\n")
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
                g.score(img, text_none)
                tap.off()
                cap[L] = tap.vec
            for L, tap in taps.items():
                tap.set(cap[L])
                rec["probs"][f"absent|a<-same@L{L}"] = g.score(img, text_hint)
                tap.off()
            if prev is not None:
                for L, tap in taps.items():
                    tap.set(prev[L])
                    rec["probs"][f"absent|a<-shuf@L{L}"] = g.score(img, text_hint)
                    tap.off()
            prev = cap
            rec["probs"]["absent|a"] = g.score(img, text_hint)
            rec["probs"]["absent|none"] = g.score(img, text_none)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
