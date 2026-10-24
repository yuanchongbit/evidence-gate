"""External validation on VizWiz val (parquet from ModelScope mirror).

Answerable vs unanswerable questions, each with / without a textual hint that
claims an answer. Mirrors the main 2x3 design with no image editing.
"""
import argparse
import glob
import io
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .infer import DATA, ROOT, SYSTEM

REFUSALS = {
    "unknown", "unanswerable", "cannot tell", "can't tell", "cant tell",
    "not enough information", "insufficient information", "unable to tell",
    "unable to determine", "not sure", "cannot determine", "can't determine",
}


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def vqa_acc(pred, answers):
    pred = norm(pred)
    hits = sum(1 for a in answers if norm(a) == pred)
    return min(hits / 3.0, 1.0)


def is_refusal(s):
    t = norm(s)
    if t in REFUSALS:
        return True
    return any(t.startswith(r) for r in ("unanswer", "cannot answer", "cant answer"))


def load_meta(ann_dir):
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(glob.glob(str(Path(ann_dir) / "val-*.parquet"))):
        t = pq.read_table(p, columns=["question_id", "question", "answers", "category"])
        rows.extend(t.to_pylist())
    return rows


def pick(meta, n_each, seed):
    rng = np.random.default_rng(seed)
    by = {0: [], 1: []}
    for r in meta:
        k = 0 if r["category"] == "unanswerable" else 1
        by[k].append(r)
    chosen = []
    for k in (0, 1):
        rows = by[k]
        idx = rng.permutation(len(rows))[: n_each]
        for i in idx:
            r = rows[i]
            ans = [a for a in (r["answers"] or []) if a]
            if not ans:
                continue
            maj = max(set(ans), key=ans.count)
            chosen.append({"sample_id": r["question_id"], "question": r["question"],
                           "answerable": k, "gold": maj, "answers": ans,
                           "category": r["category"]})
    pools = {}
    for r in chosen:
        pools.setdefault(r["category"], []).append(r["gold"])
    for cat in pools:
        pools[cat] = sorted({p for p in pools[cat]
                             if len(norm(p)) > 1 and norm(p) not in REFUSALS})
    glob = sorted({p for v in pools.values() for p in v})
    for r in chosen:
        # same-category wrong answers are more plausible (yes->no, 2->5,
        # red->blue); unanswerable items have no answer category, so use global
        cand = pools.get(r["category"]) or []
        if r["answerable"] == 0 or len(cand) < 2:
            cand = glob
        while True:
            w = cand[int(rng.integers(len(cand)))]
            if norm(w) != norm(r["gold"]):
                break
        r["wrong_hint"] = w
    return chosen


class VizScorer:
    def __init__(self, model_path, max_pixels=1024 * 1024):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa").eval()
        except TypeError:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa").eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 256, max_pixels=max_pixels)
        tok = self.processor.tokenizer
        ids = [tok.encode(" unknown", add_special_tokens=False),
               tok.encode("unknown", add_special_tokens=False)]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"unknown is not a single token: {ids}")
        self.unk_ids = [i[0] for i in ids]

    def prompt(self, text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": text}]},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def run(self, img, text):
        torch = self.torch
        p = self.prompt(text)
        batch = self.processor(text=[p], images=[img], return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch, use_cache=False, output_hidden_states=True)
        logits = out.logits[0, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        p_unk = float(max(lp[i].exp() for i in self.unk_ids))
        top = int(logits.argmax())
        maxprob = float(lp[top].exp())
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy().astype(np.float16)
        with torch.inference_mode():
            gen = self.model.generate(**batch, max_new_tokens=8, do_sample=False)
        new = gen[0, batch["input_ids"].shape[1]:]
        ans = self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()
        return {"p_unk": p_unk, "maxprob": maxprob,
                "top": self.processor.tokenizer.decode([top]),
                "gen": ans, "h": h}


def build_text(row, cond):
    q = row["question"].strip()
    if cond == "none":
        pre = ""
    elif cond == "correct":
        pre = f"Hint: the answer is {row['gold']}.\n"
    else:
        pre = f"Hint: the answer is {row['wrong_hint']}.\n"
    return f"{pre}{q}\nAnswer with one word."


def conds(row):
    return ["none", "wrong"] if row["answerable"] == 0 else ["none", "correct", "wrong"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default=str(DATA / "vizwiz_ms" / "data"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "runs" / "ext_vizwiz.jsonl"))
    ap.add_argument("--n-each", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = load_meta(args.ann_dir)
    items = pick(meta, args.n_each, args.seed)
    by_id = {r["sample_id"]: r for r in items}
    print(f"{len(items)} items "
          f"({sum(r['answerable'] for r in items)} answerable) "
          f"from {len(meta)} val questions", flush=True)
    todo = [r for r in items]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in open(out)}
        todo = [r for r in todo if r["sample_id"] not in done]
        print(f"resume: {len(done)} done, {len(todo)} left", flush=True)
    if not todo:
        return

    sc = VizScorer(args.model)
    import pyarrow.parquet as pq
    todo_ids = {r["sample_id"] for r in todo}
    t0 = time.time()
    n = 0
    keys, H = [], []
    with open(out, "a") as f:
        for p in sorted(glob.glob(str(Path(args.ann_dir) / "val-*.parquet"))):
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=16):
                d = batch.to_pylist()
                for b in d:
                    qid = b["question_id"]
                    if qid not in todo_ids:
                        continue
                    row = by_id[qid]
                    img = Image.open(io.BytesIO(b["image"]["bytes"])).convert("RGB")
                    rec = {k: row[k] for k in ("sample_id", "question", "answerable",
                                               "gold", "wrong_hint")}
                    rec["res"] = {}
                    for cond in conds(row):
                        r = sc.run(img, build_text(row, cond))
                        h = r.pop("h")
                        r["acc"] = vqa_acc(r["gen"], row["answers"])
                        r["refused"] = is_refusal(r["gen"])
                        rec["res"][cond] = r
                        keys.append(f"{qid}|{cond}")
                        H.append(h)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n += 1
                    if n % 10 == 0 or n == len(todo):
                        rate = (time.time() - t0) / n
                        print(f"{n}/{len(todo)} {rate:.2f}s/item "
                              f"eta {(len(todo)-n)*rate/60:.1f}min", flush=True)
    if H:
        np.savez_compressed(out.with_suffix(".npz"), keys=np.array(keys),
                            H=np.stack(H))
        print("hidden ->", out.with_suffix(".npz"), len(keys))


if __name__ == "__main__":
    main()
