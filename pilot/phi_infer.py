"""Phi-3.5-vision runner with the same six-condition protocol as infer.py.

Phi-3.5-vision ships its own modelling/processing code and its config pins
flash-attention-2, which is unavailable here; we load it with eager attention
and the model's native prompt format instead of a chat template.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoProcessor)

from .constants import CANDIDATES, DISTRACTOR
from .infer import ROOT, SYSTEM, build_user


def first_id(tokenizer, word):
    # Phi-3 tokenizes a leading space as its own token (29871 / the lone space
    # id, depending on tokenizer version), so take the first token that carries
    # actual text. Stripping by decoded content rather than by the id of " "
    # keeps this working when the same tokenizer reports " " differently.
    for form in (" " + word, word):
        ids = tokenizer.encode(form, add_special_tokens=False)
        while ids and not tokenizer.decode([ids[0]]).strip():
            ids = ids[1:]
        if len(ids) >= 1:
            return ids[0]
    raise ValueError(word)


class Gen:
    def __init__(self, path, resize=512, num_crops=None):
        kw = {"trust_remote_code": True}
        if num_crops is not None:
            kw["num_crops"] = num_crops
        self.proc = AutoProcessor.from_pretrained(path, **kw)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            path, config=cfg, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="cuda").eval()
        self.resize = resize
        self.system = SYSTEM
        self.ids = [first_id(self.proc.tokenizer, w) for w in CANDIDATES]
        if len(set(self.ids)) != len(CANDIDATES):
            raise RuntimeError(f"candidate ids collapsed: {self.ids}")
        print("model class:", type(self.model).__name__, flush=True)

    def prep(self, img):
        if self.resize and max(img.size) > self.resize:
            img = img.copy()
            img.thumbnail((self.resize, self.resize), Image.BICUBIC)
        return img

    def score(self, img, text, capture=None):
        prompt = ("<|system|>\n" + self.system + "<|end|>\n<|user|>\n<|image_1|>\n"
                  + text + "<|end|>\n<|assistant|>\n")
        inputs = self.proc(prompt, self.prep(img), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False,
                             output_hidden_states=capture is not None)
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits")
        s = np.array([float(logits[i]) for i in self.ids])
        s -= s.max()
        p = np.exp(s)
        p /= p.sum()
        if capture is not None:
            capture.append(out.hidden_states[-1][0, -1].float().cpu().numpy()
                           .astype(np.float16))
        return p.tolist()


def run(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = Gen(args.model, resize=args.resize, num_crops=args.num_crops)
    hidden = {}
    t0 = time.time()
    with open(out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model,
                                     "candidates": ",".join(CANDIDATES)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "probs": {}}
            acat = row.get("absent_category")
            if acat:
                rec["absent_category"] = acat
                rec["absent_hint"] = row["absent_hint"]
                rec["distractor"] = row["distractor"]
            for h in ("none", "correct", "wrong"):
                key = f"sufficient|{h}"
                buf = []
                rec["probs"][key] = g.score(img, build_user(row, h), buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            if acat:
                for tag, col in (("a", row["absent_hint"]),
                                 ("b", DISTRACTOR[row["absent_hint"]])):
                    key = f"absent|{tag}"
                    buf = []
                    rec["probs"][key] = g.score(
                        img, build_user(row, "color", acat, col), buf)
                    hidden[f"{row['sample_id']}|{key}"] = buf[0]
                key = "absent|none"
                buf = []
                rec["probs"][key] = g.score(img, build_user(row, "none", acat), buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 5 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample "
                      f"eta {(len(rows) - n) * rate / 60:.1f}min", flush=True)
    keys = list(hidden)
    np.savez_compressed(out.with_suffix(".npz"), keys=np.array(keys),
                        H=np.stack([hidden[k] for k in keys]))
    print("saved", out, np.stack([hidden[k] for k in keys]).shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/phi.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--num-crops", type=int, default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
