"""Prompt-format robustness: no-unknown instruction and conversational hint.

Every prompt in the main battery offers an explicit unknown option and a
bracketed "Hint:" prefix, which may overstate abstention and hint uptake. This
runs the same scenes three ways -- the main format, a system prompt with no
unknown option, and a hint phrased as a sentence -- and saves final states so
the frozen probe can be tested on the new formats.
"""

import argparse
import json
import time

import numpy as np
from PIL import Image

from .constants import DISTRACTOR
from .infer import DATA, ROOT, SYSTEM, Scorer, build_user
from .infer_gen import Gen

NOUNK_SYSTEM = (
    "You are a careful visual question answering assistant. Answer with exactly "
    "one word, which must be a colour."
)


def nl_hint(row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    if hint == "correct":
        prefix = f"A colleague mentioned that the {cat} is {row['gold']}. "
    elif hint == "wrong":
        prefix = f"A colleague mentioned that the {cat} is {DISTRACTOR[row['gold']]}. "
    elif hint == "color":
        prefix = f"A colleague mentioned that the {cat} is {hint_color}. "
    else:
        prefix = ""
    return f"{prefix}What color is the {cat} in this photo?\nAnswer with one word."


class SysScorer(Scorer):
    def __init__(self, model_path, system):
        self.system = system
        super().__init__(model_path)

    def _inputs(self, images, text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system}]},
            {"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": text}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt] * len(images), images=images,
                               padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}


class SysGen(Gen):
    def __init__(self, path, system, resize=512):
        self.sys = system
        super().__init__(path, resize=resize)

    def inputs(self, images, text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.sys}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
        prompt = self.proc.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        batch = self.proc(text=[prompt] * len(images), images=list(images),
                          padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--family", default="qwen", choices=["qwen", "gen"])
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "runs" / "prompt_variants.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")]
    if args.limit:
        rows = rows[: args.limit]

    if args.family == "qwen":
        std = SysScorer(args.model, SYSTEM)

        def score(m, img, text, buf=None):
            return m.score([img], text, capture=buf)[0]
    else:
        std = SysGen(args.model, SYSTEM, resize=args.resize)

        def score(m, img, text, buf=None):
            return m.score(img, text, capture=buf)
    cand = std.cand_names if args.family == "qwen" else list(std.ids)

    def set_system(fmt):
        text = NOUNK_SYSTEM if fmt == "nounk" else SYSTEM
        if args.family == "qwen":
            std.system = text
        else:
            std.sys = text

    def prompts(row):
        acat, acolor = row["absent_category"], row["absent_hint"]
        out = []
        for fmt, fn in (("std", build_user), ("nounk", build_user), ("nl", nl_hint)):
            out.append((f"{fmt}|absent|none", fn(row, "none", acat)))
            for tag, col in (("a", acolor), ("b", DISTRACTOR[acolor])):
                out.append((f"{fmt}|absent|{tag}", fn(row, "color", acat, col)))
            out.append((f"{fmt}|sufficient|wrong", fn(row, "wrong")))
            out.append((f"{fmt}|sufficient|none", fn(row, "none")))
        return out

    hidden = {}
    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "family": args.family,
                                     "resize": args.resize,
                                     "candidates": ",".join(cand)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for key, text in prompts(row):
                set_system(key.split("|", 1)[0])
                buf = []
                rec["probs"][key] = score(std, img, text, buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"),
                        keys=np.array(keys), H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out, "hidden", len(keys))


if __name__ == "__main__":
    main()
