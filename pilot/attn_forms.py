"""Attention routing across prompt forms (Qwen2.5-VL, eager attention).

For a subset of scenes and each prompt form, records the attention share that
the final query position assigns to the assertion tokens, the image tokens and
the question tokens. The question is whether the framing that produces hint
adoption is also the framing in which the final readout looks at the assertion
instead of the image.
"""

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .constants import CANDIDATES, DISTRACTOR
from .infer import ROOT, SYSTEM
from .prompt_forms import FORMS, NEUTRAL_TURN, question, user_text

CELLS = (("absent", "a"), ("sufficient", "wrong"))


def build_messages(form, row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    q = question(cat)
    text = user_text(form, row, hint, cat, hint_color)
    system = SYSTEM
    turn = NEUTRAL_TURN
    if hint not in ("none",):
        color = (row["gold"] if hint == "correct" else
                 DISTRACTOR[row["gold"]] if hint == "wrong" else hint_color)
        if form == "system":
            system = SYSTEM + " " + f"The {cat} is {color}."
        elif form == "dialogue":
            turn = f"I think the {cat} is {color}."
    if form == "dialogue":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": turn}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Okay."}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
    return messages, system, turn, text


def assertion_text(form, row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    if hint == "correct":
        color = row["gold"]
    elif hint == "wrong":
        color = DISTRACTOR[row["gold"]]
    else:
        color = hint_color
    if form == "nl":
        return f"A colleague mentioned that the {cat} is {color}."
    if form == "claim":
        return f"The {cat} is {color}."
    if form == "belief":
        return f"I think the {cat} is {color}."
    if form == "system":
        return f"The {cat} is {color}."
    if form == "dialogue":
        return f"I think the {cat} is {color}."
    return f"Hint: the {cat} is {color}."


def span_of(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if a < end and b > start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest_v5.jsonl"))
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3b"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--forms", default=",".join(FORMS))
    ap.add_argument("--out", default=str(ROOT / "runs" / "attn_forms_3b.jsonl"))
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 256,
                                         max_pixels=1024 * 1024)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    tok = proc.tokenizer
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")
    cand_ids = {}
    for w in CANDIDATES:
        for form in (" " + w, w):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                cand_ids[w] = ids[0]
                break

    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r.get("absent_category")][:: args.stride]
    if args.limit:
        rows = rows[: args.limit]
    forms = args.forms.split(",")

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"model": args.model, "forms": forms,
                                     "stride": args.stride}}) + "\n")
        for n, row in enumerate(rows, 1):
            img = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            if max(img.size) > 512:
                img = img.copy()
                img.thumbnail((512, 512), Image.BICUBIC)
            for form in forms:
                for ev, hint in CELLS:
                    cat = row["absent_category"] if ev == "absent" else None
                    col = (row["absent_hint"] if ev == "absent" else None)
                    messages, system, turn, text = build_messages(
                        form, row, hint, cat, col)
                    prompt = proc.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    enc = tok(prompt, return_offsets_mapping=True,
                              add_special_tokens=False)
                    ids = enc["input_ids"]
                    offs = enc["offset_mapping"]
                    inputs = proc(text=[prompt], images=[img],
                                  return_tensors="pt").to("cuda")
                    full = inputs["input_ids"][0].tolist()
                    diff = next((i for i in range(min(len(ids), len(full)))
                                 if ids[i] != full[i]), len(ids))
                    n_img = 0
                    j = diff
                    while j < len(full) and full[j] == img_id:
                        n_img += 1
                        j += 1
                    shift = n_img - 1
                    a_txt = assertion_text(form, row, hint, cat, col)
                    q_txt = question(cat or row["category"])
                    ai = prompt.find(a_txt)
                    qi = prompt.find(q_txt)
                    if ai < 0 or qi < 0:
                        raise RuntimeError(f"span not found: {form} {hint}")
                    a_idx = [i if i < diff else i + shift
                             for i in span_of(offs, ai, ai + len(a_txt))]
                    q_idx = [i if i < diff else i + shift
                             for i in span_of(offs, qi, qi + len(q_txt))]
                    img_idx = list(range(diff, diff + n_img))
                    with torch.inference_mode():
                        out = model(**inputs, use_cache=False,
                                    output_attentions=True)
                    att = torch.stack(out.attentions)[:, 0, :, -1, :].float()
                    wl = att.mean(dim=1).cpu().numpy()
                    w = wl.mean(axis=0)
                    tot = float(w.sum())
                    logits = out.logits[0, -1].float().cpu().numpy()
                    pred = max(cand_ids, key=lambda k: logits[cand_ids[k]])
                    rec = {
                        "sample_id": row["sample_id"], "form": form,
                        "ev": ev, "hint": hint,
                        "hint_share": float(w[a_idx].sum() / tot),
                        "image_share": float(w[img_idx].sum() / tot),
                        "question_share": float(w[q_idx].sum() / tot),
                        "hint_share_layers": [
                            float(wl[l][a_idx].sum() / wl[l].sum())
                            for l in range(wl.shape[0])],
                        "image_share_layers": [
                            float(wl[l][img_idx].sum() / wl[l].sum())
                            for l in range(wl.shape[0])],
                        "n_hint": len(a_idx), "n_img": len(img_idx),
                        "n_tokens": len(full),
                        "pred": pred,
                        "stated": (row["absent_hint"] if ev == "absent"
                                   else DISTRACTOR[row["gold"]]),
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    del out, att, w
                    torch.cuda.empty_cache()
            if n % 5 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/scene "
                      f"eta {(len(rows) - n) * rate / 60:.1f}min", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
