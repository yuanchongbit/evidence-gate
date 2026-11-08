"""Merge the neutral-turn dialogue no-hint cells into the Gemma battery."""

import json
import sys
from pathlib import Path

import numpy as np


def main(main_jsonl, fix_jsonl, main_npz, fix_npz, out_stem):
    records = [json.loads(line) for line in open(main_jsonl)]
    meta, rows = records[0], records[1:]
    fix = {r["sample_id"]: r for r in
           (json.loads(line) for line in open(fix_jsonl)) if "probs" in r}
    cells = ("dialogue|absent|none", "dialogue|sufficient|none")
    n = 0
    for row in rows:
        fx = fix.get(row["sample_id"])
        if fx is None:
            continue
        for cell in cells:
            row["probs"][cell] = fx["probs"][cell]
        n += 1
    out_jsonl = str(Path(out_stem).with_suffix(".jsonl"))
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")

    hm = np.load(main_npz, allow_pickle=False)
    hf = np.load(fix_npz, allow_pickle=False)
    fix_map = dict(zip(hf["keys"], hf["H"]))
    keys = list(hm["keys"])
    H = hm["H"].copy()
    replaced = 0
    for i, key in enumerate(keys):
        if key in fix_map:
            H[i] = fix_map[key]
            replaced += 1
    np.savez_compressed(str(Path(out_stem).with_suffix(".npz")),
                        keys=np.array(keys), H=H)
    print(f"merged {n} scenes, replaced {replaced}/{len(keys)} cells")


if __name__ == "__main__":
    main(*sys.argv[1:6])
