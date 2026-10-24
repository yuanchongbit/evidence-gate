"""Model-agnostic version of infer.py for cross-family replication.

Uses AutoModelForImageTextToText so the same manifest / prompt / candidate
readout can be run on LLaVA-OneVision, Idefics3/SmolVLM, etc. Output format
matches runs/*.jsonl + runs/*.npz consumed by probe.py / baselines.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .constants import CANDIDATES, DISTRACTOR
from .infer import ROOT, SYSTEM, build_user


def first_id(tokenizer, word):
    for form in (" " + word, word):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) >= 1:
            return ids[0]
    raise ValueError(word)


class Gen:
    def __init__(self, path, max_pixels=1024 * 1024, resize=1024):
        kw = dict(min_pixels=256 * 256, max_pixels=max_pixels)
        try:
            self.proc = AutoProcessor.from_pretrained(path, **kw)
        except TypeError:
            self.proc = AutoProcessor.from_pretrained(path)
        loaders = [
            dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"),
            dict(torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"),
        ]
        for kwm in loaders:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(path, **kwm).eval()
                break
            except TypeError:
                continue
        self.resize = resize
        self.ids = {w: first_id(self.proc.tokenizer, w) for w in CANDIDATES}
        print("model class:", type(self.model).__name__, flush=True)

    def prep(self, img):
        if self.resize and max(img.size) > self.resize:
            img = img.copy()
            img.thumbnail((self.resize, self.resize), Image.BICUBIC)
        return img

    def inputs(self, images, text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
        prompt = self.proc.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        batch = self.proc(text=[prompt] * len(images), images=list(images),
                          padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}

    def score(self, img, text, capture=None):
        inputs = self.inputs([self.prep(img)], text)
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False, output_hidden_states=capture is not None)
        logits = out.logits[0, -1].float()
        s = np.array([float(logits[i]) for i in [self.ids[w] for w in CANDIDATES]])
        s -= s.max()
        p = np.exp(s)
        p /= p.sum()
        if capture is not None:
            capture.append(out.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float16))
        return p.tolist()


def run(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = Gen(args.model, resize=args.resize)
    hidden = {}
    t0 = time.time()
    with open(out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "candidates": ",".join(CANDIDATES)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "probs": {}}
            acat = row.get("absent_category")
            if acat:
                rec["absent_category"] = acat
                rec["absent_hint"] = row["absent_hint"]
            for h in ("none", "correct", "wrong"):
                key = f"sufficient|{h}"
                buf = []
                rec["probs"][key] = g.score(img, build_user(row, h), buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            if acat:
                for tag, col in (("a", row["absent_hint"]), ("b", DISTRACTOR[row["absent_hint"]])):
                    key = f"absent|{tag}"
                    buf = []
                    rec["probs"][key] = g.score(img, build_user(row, "color", acat, col), buf)
                    hidden[f"{row['sample_id']}|{key}"] = buf[0]
                key = "absent|none"
                buf = []
                rec["probs"][key] = g.score(img, build_user(row, "none", acat), buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 5 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample eta {(len(rows)-n)*rate/60:.1f}min",
                      flush=True)
    keys = list(hidden)
    np.savez_compressed(out.with_suffix(".npz"), keys=np.array(keys),
                        H=np.stack([hidden[k] for k in keys]))
    print("saved", out, np.stack([hidden[k] for k in keys]).shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "gen.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resize", type=int, default=768)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
