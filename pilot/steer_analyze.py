"""Summarise the steering sweep: does moving along the probe direction move
the model's answer/refuse decision in the predicted way?"""

import argparse
import json
import statistics

import numpy as np

mean = statistics.mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/steer_261.jsonl")
    args = ap.parse_args()
    lines = [json.loads(l) for l in open(args.run)]
    meta = lines[0]["meta"]
    recs = [r for r in lines if "meta" not in r]
    names = meta["candidates"].split(",")
    settings = meta["alphas"].split(",")
    order = ["base", "ablate"] + [f"a{float(a):+.2f}" for a in settings]

    for cond, tag in (("sufficient|none", "present"), ("absent|none", "absent")):
        print(f"\n== {tag} ({cond})  n={len(recs)}")
        print("setting   refuse  p_unknown   top1_conf   answers_colour")
        for s in order:
            key = f"{cond}|{s}"
            if key not in recs[0]["probs"]:
                continue
            pr = np.array([r["probs"][key] for r in recs])
            top = pr.argmax(1)
            refuse = float((top == names.index("unknown")).mean())
            pun = float(mean(pr[:, names.index("unknown")]))
            conf = float(mean(pr.max(1)))
            print(f"{s:8s}  {refuse:5.3f}   {pun:8.3f}   {conf:9.3f}   {1 - refuse:5.3f}")


if __name__ == "__main__":
    main()
