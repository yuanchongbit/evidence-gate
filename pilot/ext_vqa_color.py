"""External replication of the colour task on VQA-v2 val questions whose gold
answer is one of our 11 colours. Images are COCO val2017 (never used to build
the 261-sample set). The probe is frozen: trained on the 261-sample set, then
applied here without refitting.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CANDIDATES, COLORS, DISTRACTOR
from .infer import DATA, ROOT, Scorer
from .metrics import auroc
from .probe import cv_oof, dataset, load

PAT = re.compile(r"what colou?r (?:is|are) (?:the |this |that |these |those )?(.+?)\s*\??$",
                 re.I)


def build(obj, gold, hint):
    if hint == "none":
        pre = ""
    elif hint == "correct":
        pre = f"Hint: the {obj} is {gold}.\n"
    else:
        pre = f"Hint: the {obj} is {DISTRACTOR[gold]}.\n"
    return f"{pre}What color is the {obj}?\nAnswer with one word."


def load_vqa(qpath, apath, img_dir, n, seed):
    qs = {q["question_id"]: q for q in json.load(open(qpath))["questions"]}
    anns = json.load(open(apath))["annotations"]
    cand = []
    for a in anns:
        gold = a["multiple_choice_answer"].lower()
        if gold not in COLORS:
            continue
        q = qs[a["question_id"]]["question"].strip()
        m = PAT.search(q)
        if not m:
            continue
        obj = m.group(1).strip().lower()
        if not obj or len(obj.split()) > 4:
            continue
        img = Path(img_dir) / f"{a['image_id']:012d}.jpg"
        if not img.exists():
            continue
        cand.append({"sample_id": str(a["question_id"]), "image": str(img),
                     "image_id": a["image_id"], "question": q, "obj": obj,
                     "gold": gold,
                     "answers": [x["answer"] for x in a["answers"]]})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cand))[:n]
    return [cand[i] for i in sorted(idx)]


def fit_probe(run, Cs):
    recs, meta, H = load(run)
    X, y, g, _, _ = dataset(recs, H, "sufficient|none", "absent|none", "sample")
    best = None
    for C in Cs:
        oof = cv_oof(X, y, g, C)
        pt = auroc(oof, y)
        if best is None or pt > best[0]:
            best = (pt, C)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000))
    clf.fit(X, y)
    return clf, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(DATA / "vqa2" / "v2_OpenEnded_mscoco_val2014_questions.json"))
    ap.add_argument("--annotations", default=str(DATA / "vqa2" / "v2_mscoco_val2014_annotations.json"))
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--probe-run", default=str(ROOT / "runs" / "v5_261.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "ext_vqa_color.jsonl"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_vqa(args.questions, args.annotations, args.images, args.n, args.seed)
    print(f"{len(items)} colour questions", flush=True)
    clf, (pt, C) = fit_probe(args.probe_run, [0.0003, 0.001, 0.003, 0.01, 0.1])
    print(f"frozen probe: AUROC {pt:.3f} on 261 set, C={C:g}", flush=True)

    sc = Scorer(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in open(out)}
        items = [r for r in items if r["sample_id"] not in done]
        print(f"resume: {len(done)} done", flush=True)

    keys, H = [], []
    t0 = time.time()
    with open(out, "a") as f:
        for n, row in enumerate(items, 1):
            img = Image.open(row["image"]).convert("RGB")
            rec = {k: row[k] for k in ("sample_id", "image_id", "question", "obj", "gold")}
            rec["hint_wrong"] = DISTRACTOR[row["gold"]]
            rec["res"] = {}
            for hint in ("none", "correct", "wrong"):
                buf = []
                p = sc.score([img], build(row["obj"], row["gold"], hint),
                             batch_size=1, capture=buf)[0]
                h = buf[0].astype(np.float32)
                s = float(clf.decision_function(h[None, :])[0])
                rec["res"][hint] = {
                    "p_unk": float(p[-1]),
                    "maxprob": float(np.max(p)),
                    "pred": CANDIDATES[int(np.argmax(p))],
                    "probe": s,
                }
                keys.append(f"{row['sample_id']}|{hint}")
                H.append(h.astype(np.float16))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(items):
                rate = (time.time() - t0) / n
                print(f"{n}/{len(items)} {rate:.2f}s/item "
                      f"eta {(len(items)-n)*rate/60:.1f}min", flush=True)
    np.savez_compressed(out.with_suffix(".npz"), keys=np.array(keys), H=np.stack(H))
    print("saved", out)


if __name__ == "__main__":
    main()
