"""Naturalistic prompt forms for the hint manipulation (Qwen family).

Forms: std (bracketed Hint:), nl (colleague report), claim (direct statement),
belief (first-person), system (assertion in the system message), dialogue
(two-turn first-person assertion).

Every form is run with absent none/a/b and sufficient none/wrong, so behaviour
and the evidence-axis geometry can be compared within form. The dialogue
no-hint cell keeps the same two-turn structure with a neutral first turn.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CANDIDATES, DISTRACTOR
from .infer import DATA, ROOT, SYSTEM, Scorer

FORMS = ("std", "nl", "claim", "belief", "system", "dialogue")
NEUTRAL_TURN = "Let's look at this photo."


def question(cat):
    return f"What color is the {cat}?\nAnswer with one word."


def assertion(cat, color):
    return f"The {cat} is {color}."


def user_text(form, row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    q = question(cat)
    if hint == "none":
        return q
    if hint == "correct":
        color = row["gold"]
    elif hint == "wrong":
        color = DISTRACTOR[row["gold"]]
    else:
        color = hint_color
    if form == "std":
        return f"Hint: the {cat} is {color}.\n{q}"
    if form == "nl":
        return f"A colleague mentioned that the {cat} is {color}. {q}"
    if form == "claim":
        return f"The {cat} is {color}. {q}"
    if form == "belief":
        return f"I think the {cat} is {color}. {q}"
    return q


class FormScorer(Scorer):
    def __init__(self, model_path, dtype="bfloat16"):
        self.form = "std"
        self.system = SYSTEM
        self.turn = ""
        super().__init__(model_path, dtype=dtype)

    def _inputs(self, images, text):
        if self.form == "dialogue":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user", "content": [{"type": "text", "text": self.turn or NEUTRAL_TURN}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Okay."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            ]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt] * len(images), images=images,
                               padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}


def set_form(scorer, form, row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    scorer.form = form
    scorer.system = SYSTEM
    scorer.turn = ""
    if hint == "none":
        return
    if hint == "correct":
        color = row["gold"]
    elif hint == "wrong":
        color = DISTRACTOR[row["gold"]]
    else:
        color = hint_color
    if form == "system":
        scorer.system = SYSTEM + " " + assertion(cat, color)
    elif form == "dialogue":
        scorer.turn = f"I think the {cat} is {color}."


def prompts(row):
    acat, acolor = row["absent_category"], row["absent_hint"]
    out = []
    for form in FORMS:
        out.append((form, "absent", "none", acat, None))
        for tag, col in (("a", acolor), ("b", DISTRACTOR[acolor])):
            out.append((form, "absent", tag, acat, col))
        out.append((form, "sufficient", "none", None, None))
        out.append((form, "sufficient", "wrong", None, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "prompt_forms.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [row for row in rows if row.get("absent_category")][:: args.stride]
    if args.limit:
        rows = rows[: args.limit]

    model = FormScorer(args.model)
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
                rec["probs"][f"{form}|{ev}|{hint}"] = model.score([img], text, capture=buf)[0]
                hidden[f"{row['sample_id']}|{form}|{ev}|{hint}"] = buf[0]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample "
                      f"eta {(len(rows) - n) * rate / 60:.1f}min", flush=True)
    keys = list(hidden)
    np.savez_compressed(args.out.replace(".jsonl", ".npz"),
                        keys=np.array(keys), H=np.stack([hidden[k] for k in keys]))
    print("saved", args.out, "hidden", len(keys))


if __name__ == "__main__":
    main()
