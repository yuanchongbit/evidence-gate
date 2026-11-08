"""Six-form battery on the model-agnostic Gen backend (Gemma-3).

Same cells as prompt_forms.py plus the agreeing-hint cell, so behaviour and
evidence-axis geometry can be compared across prompt forms on a second model
family.  Hidden states are captured at the final position, keyed
sample|form|ev|hint, matching prompt_forms_analyze.load_states.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CANDIDATES, DISTRACTOR
from .infer import DATA, ROOT, SYSTEM
from .infer_gen import Gen
from .prompt_forms import FORMS, NEUTRAL_TURN, user_text


class FormGen(Gen):
    def __init__(self, path, resize=512, dtype="bfloat16"):
        self.form = "std"
        self.sysmsg = SYSTEM
        self.turn = ""
        super().__init__(path, resize=resize, dtype=dtype)

    def inputs(self, images, text):
        sys_msg = {"role": "system",
                   "content": [{"type": "text", "text": self.sysmsg}]}
        img = {"type": "image"}
        if self.form == "dialogue":
            messages = [sys_msg,
                        {"role": "user",
                         "content": [{"type": "text", "text": self.turn}]},
                        {"role": "assistant",
                         "content": [{"type": "text", "text": "Okay."}]},
                        {"role": "user",
                         "content": [img, {"type": "text", "text": text}]}]
        else:
            messages = [sys_msg,
                        {"role": "user",
                         "content": [img, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.proc(text=[prompt] * len(images), images=list(images),
                          padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}


def set_form(model, form, row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    model.form, model.sysmsg, model.turn = form, SYSTEM, ""
    if hint == "none":
        return
    if hint == "correct":
        color = row["gold"]
    elif hint == "wrong":
        color = DISTRACTOR[row["gold"]]
    else:
        color = hint_color
    if form == "system":
        model.sysmsg = SYSTEM + " " + f"The {cat} is {color}."
    elif form == "dialogue":
        model.turn = f"I think the {cat} is {color}."


def prompts(row):
    acat, acolor = row["absent_category"], row["absent_hint"]
    out = []
    for form in FORMS:
        out.append((form, "absent", "none", acat, None))
        for tag, col in (("a", acolor), ("b", DISTRACTOR[acolor])):
            out.append((form, "absent", tag, acat, col))
        out.append((form, "sufficient", "none", None, None))
        out.append((form, "sufficient", "correct", None, None))
        out.append((form, "sufficient", "wrong", None, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", default=str(ROOT / "models" / "gemma3_4b"))
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "runs" / "prompt_forms_gen.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")][:: args.stride]
    if args.limit:
        rows = rows[: args.limit]

    model = FormGen(args.model, resize=args.resize)
    hidden = {}
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "forms": list(FORMS),
                                     "stride": args.stride,
                                     "candidates": ",".join(CANDIDATES)}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "absent_category": row["absent_category"],
                   "absent_hint": row["absent_hint"], "probs": {}}
            for form, ev, hint, cat, col in prompts(row):
                set_form(model, form, row, hint, cat, col)
                text = user_text(form, row, hint, cat, col)
                buf = []
                key = f"{form}|{ev}|{hint}"
                rec["probs"][key] = model.score(img, text, capture=buf)
                hidden[f"{row['sample_id']}|{key}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 5 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample "
                      f"eta {(len(rows) - n) * rate / 60:.1f}min", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"),
                        keys=np.array(keys), H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out, "hidden", len(keys))


if __name__ == "__main__":
    main()
