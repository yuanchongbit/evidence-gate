import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import CANDIDATES, DISTRACTOR, HINTS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SYSTEM = (
    "You are a careful visual question answering assistant. "
    "Answer with exactly one word. "
    'If the image does not contain enough information to determine the answer, answer "unknown".'
)


def build_user(row, hint, category=None, hint_color=None):
    cat = category or row["category"]
    if hint == "correct":
        prefix = f"Hint: the {cat} is {row['gold']}.\n"
    elif hint == "wrong":
        prefix = f"Hint: the {cat} is {DISTRACTOR[row['gold']]}.\n"
    elif hint == "color":
        prefix = f"Hint: the {cat} is {hint_color}.\n"
    else:
        prefix = ""
    return f"{prefix}What color is the {cat}?\nAnswer with one word."


def grid_boxes(width, height, n):
    xs = np.linspace(0, width, n + 1).astype(int)
    ys = np.linspace(0, height, n + 1).astype(int)
    return [(int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
            for j in range(n) for i in range(n)]


def occlude(img, box):
    """Fill a region with the local ring median so occluding background cells
    barely changes the image while occluding the object removes it."""
    arr = np.asarray(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    x1, y1 = min(w, x1 + 2), min(h, y1 + 2)
    if x1 <= x0 or y1 <= y0:
        return Image.fromarray(arr)
    bw, bh = x1 - x0, y1 - y0
    pad = max(4, int(0.15 * max(bw, bh)))
    ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
    ax1, ay1 = min(w, x1 + pad), min(h, y1 + pad)
    area = arr[ay0:ay1, ax0:ax1]
    inside = np.zeros(area.shape[:2], bool)
    inside[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] = True
    ring = area[~inside]
    med = np.median(ring, axis=0) if len(ring) else area.reshape(-1, 3).mean(0)

    rng = np.random.default_rng(0)
    fill = med + rng.normal(0, 3.0, (bh, bw, 3))

    f = max(1, min(3, min(bw, bh) // 4))
    alpha = np.ones((bh, bw), np.float32)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    alpha[:f, :] *= ramp[:, None]
    alpha[-f:, :] *= ramp[::-1, None]
    alpha[:, :f] *= ramp[None, :]
    alpha[:, -f:] *= ramp[::-1][None, :]
    block = arr[y0:y1, x0:x1].astype(np.float32)
    arr[y0:y1, x0:x1] = np.clip(
        alpha[..., None] * fill + (1.0 - alpha[..., None]) * block, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr)


def bbox_box(row, width, height):
    x, y, w, h = row["bbox"]
    return (max(0, int(x)), max(0, int(y)), min(width, int(x + w)), min(height, int(y + h)))


class Scorer:
    def __init__(self, model_path, dtype="bfloat16", max_pixels=1024 * 1024):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        kwargs = dict(device_map="cuda", attn_implementation="sdpa")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, dtype=getattr(torch, dtype), **kwargs
            ).eval()
        except TypeError:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=getattr(torch, dtype), **kwargs
            ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 256, max_pixels=max_pixels
        )
        self.cand_ids = self._candidate_ids()
        self.cand_names = CANDIDATES

    def _candidate_ids(self):
        tok = self.processor.tokenizer
        ids = {}
        for word in CANDIDATES:
            forms = [
                tok.encode(" " + word, add_special_tokens=False),
                tok.encode(word, add_special_tokens=False),
            ]
            single = [f[0] for f in forms if len(f) == 1]
            if not single:
                raise ValueError(f"candidate {word!r} is not a single token: "
                                 f"{[tok.convert_ids_to_tokens(f) for f in forms]}")
            ids[word] = sorted(set(single))
        return ids

    def _inputs(self, images, text):
        torch = self.torch
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": text}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[prompt] * len(images), images=images, padding=True, return_tensors="pt"
        )
        return {k: v.to("cuda") for k, v in batch.items()}

    def score(self, images, text, batch_size=8, capture=None):
        torch = self.torch
        probs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            inputs = self._inputs(chunk, text)
            with torch.inference_mode():
                out = self.model(**inputs, use_cache=False, output_hidden_states=capture is not None)
                logits = out.logits[:, -1, :].float()
                if capture is not None:
                    h = out.hidden_states[-1][:, -1, :].float().cpu().numpy().astype(np.float16)
                    capture.extend(h)
            for row in logits:
                scores = []
                for word in self.cand_names:
                    scores.append(max(row[j].item() for j in self.cand_ids[word]))
                scores = np.array(scores)
                scores -= scores.max()
                p = np.exp(scores)
                probs.append((p / p.sum()).tolist())
        return probs


def run(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.samples:
        keep = set(args.samples.split(","))
        rows = [r for r in rows if r["sample_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(l)["sample_id"] for l in open(out_path)}
        rows = [r for r in rows if r["sample_id"] not in done]
        print(f"resume: {len(done)} done, {len(rows)} left")

    if args.dry_run:
        r = rows[0]
        print("system:", SYSTEM)
        for h in HINTS:
            print(f"[{h}]", build_user(r, h))
        print("conditions per sample:", 2 * len(HINTS), "+ grid", 2 * args.grid ** 2,
              "+ bbox 2 + cf 2")
        print("forward calls per sample:", 2 * len(HINTS) + 2 * args.grid ** 2 + 4)
        return

    scorer = Scorer(args.model, dtype=args.dtype)
    order = ",".join(scorer.cand_names)
    if not done:
        with open(out_path, "w") as f:
            f.write(json.dumps({"meta": {"model": args.model, "candidates": order,
                                         "grid": args.grid, "probe_hint": "none",
                                         "dtype": args.dtype}}) + "\n")

    t0 = time.time()
    hidden = {}
    with open(out_path, "a") as f:
        for n, row in enumerate(rows, 1):
            orig = Image.open(ROOT / row["renders"]["orig"]).convert("RGB")
            cf_t = cf_p = None
            if not args.skip_probes:
                cf_t = Image.open(ROOT / row["renders"]["cf_target"]).convert("RGB")
                cf_p = Image.open(ROOT / row["renders"]["cf_placebo"]).convert("RGB")

            rec = {"sample_id": row["sample_id"], "category": row["category"],
                   "gold": row["gold"], "distractor": row["distractor"],
                   "cf_color": row.get("cf_color"), "probs": {}, "occ": {}, "occ_gt": {},
                   "cf": {}}
            if row.get("absent_category"):
                rec["absent_category"] = row["absent_category"]
                rec["absent_hint"] = row["absent_hint"]

            imgs = [("sufficient", orig)]
            if args.with_insufficient:
                ins = Image.open(ROOT / row["renders"]["insufficient"]).convert("RGB")
                imgs.append(("insufficient", ins))
            for ev, img in imgs:
                for h in HINTS:
                    key = f"{ev}|{h}"
                    buf = [] if args.save_hidden else None
                    rec["probs"][key] = scorer.score(
                        [img], build_user(row, h), batch_size=1, capture=buf)[0]
                    if buf is not None:
                        hidden[f"{row['sample_id']}|{key}"] = buf[0]

            if row.get("absent_category"):
                acat, acolor = row["absent_category"], row["absent_hint"]
                alt = DISTRACTOR[acolor]
                for tag, col in (("a", acolor), ("b", alt)):
                    key = f"absent|{tag}"
                    buf = [] if args.save_hidden else None
                    rec["probs"][key] = scorer.score(
                        [orig], build_user(row, "color", acat, col),
                        batch_size=1, capture=buf)[0]
                    if buf is not None:
                        hidden[f"{row['sample_id']}|{key}"] = buf[0]
                key = "absent|none"
                buf = [] if args.save_hidden else None
                rec["probs"][key] = scorer.score(
                    [orig], build_user(row, "none", acat), batch_size=1, capture=buf)[0]
                if buf is not None:
                    hidden[f"{row['sample_id']}|{key}"] = buf[0]

            if not args.skip_probes:
                for ev, img in imgs:
                    boxes = grid_boxes(img.width, img.height, args.grid)
                    occ_imgs = [occlude(img, b) for b in boxes]
                    rec["occ"][ev] = scorer.score(occ_imgs, build_user(row, "none"))
                    gt = occlude(img, bbox_box(row, img.width, img.height))
                    rec["occ_gt"][ev] = scorer.score([gt], build_user(row, "none"))[0]

                rec["cf"]["target"] = scorer.score([cf_t], build_user(row, "none"))[0]
                rec["cf"]["placebo"] = scorer.score([cf_p], build_user(row, "none"))[0]

            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 5 == 0 or n == len(rows):
                rate = (time.time() - t0) / n
                print(f"{n}/{len(rows)} {rate:.1f}s/sample eta {(len(rows)-n)*rate/60:.1f}min",
                      flush=True)
    if hidden:
        hpath = out_path.with_suffix(".npz")
        keys = list(hidden)
        np.savez_compressed(hpath, keys=np.array(keys),
                            H=np.stack([hidden[k] for k in keys]))
        print("hidden states ->", hpath, np.stack([hidden[k] for k in keys]).shape)
    print("saved", out_path)


def download(model_id):
    from modelscope import snapshot_download
    path = snapshot_download(model_id)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest_v5.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "qwen25vl3b.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-probes", action="store_true",
                    help="only run the hint conditions (cheap)")
    ap.add_argument("--with-insufficient", action="store_true",
                    help="also run the erased-object condition (legacy)")
    ap.add_argument("--save-hidden", action="store_true",
                    help="save final-layer hidden state at the answer position to <out>.npz")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download(args.model)
    else:
        run(args)


if __name__ == "__main__":
    main()
