# The Evidence Gate

**Conflicting hints make vision-language models answer when the evidence is absent, but they do not erase the evidence inside the model.**

This repository contains the code, the frozen benchmark and the result summaries for the paper *The Evidence Gate: How Vision-Language Models Decide to Answer Without Evidence* (preprint in preparation).

Project page: <https://yuanchongbit.github.io/evidence-gate/> (English) · [中文](https://yuanchongbit.github.io/evidence-gate/index.zh.html)

## The question

A VLM is asked *"What color is the umbrella?"* when there is no umbrella in the image. It should say **unknown**. It usually does — until a sentence in the prompt says *"Hint: the umbrella is black."* Then it answers *black*.

Is that a failure of **seeing** or of **reporting what was seen**? We separate the two with a 2×3 within-subject design and find a dissociation:

- **Output-space evidence signals collapse under conflict.** The AUROC of p(unknown) for detecting whether the queried object is present drops from **0.993 to 0.834**; max-probability drops from 0.615 to **0.463**, below chance.
- **The internal evidence representation does not.** A linear probe on the final-token hidden state scores **0.995 / 0.998 / 0.997** under no / conflicting / correct hints, and transfers from no-hint to conflict at **0.969**.
- **The suppression is a late write.** Pasting the no-hint state back at layer 32/36 recovers **80%** of the lost refusal (3B); at 26/28 it recovers **90%** (7B); a LLaVA-OneVision model, second family, recovers **36% / 80% / 96%** at 64% / 86% / 93% depth.
- **The gate is causal and readable.** Adding or ablating one probe direction changes abstention bidirectionally and dose-monotonically on held-out halves.
- **A training-free gate works where it matters.** At 50% coverage under conflicting hints, a probe-based gate reduces fabrication from **0.249** (p(unknown)) to **0.011** (3B), and similarly on 7B and LLaVA.

The effect is not a property of one checkpoint: the same depth profile appears on Qwen2.5-VL-3B, Qwen2.5-VL-7B and LLaVA-OneVision-7B.

## Design

|  | no hint | correct hint | conflicting hint |
|---|---|---|---|
| **object present** | `sufficient\|none` | `sufficient\|correct` | `sufficient\|wrong` |
| **object verified absent** | `absent\|none` | `absent\|a` | `absent\|b` |

Every image appears in all six cells. The absent arm asks about an object category that COCO's exhaustive val2017 annotations place outside the image; the hinted colour is that category's most frequent colour in COCO val, so the hint is plausible rather than adversarial. **No pixel is edited.** A separate yes/no presence question confirms the manipulation (p(yes) = 0.095 for absent categories vs. 0.95 for the original images).

Data: 261 human-verified colour labels, 74 absent categories, 11 colours + `unknown`, COCO val2017 images. Labels were proposed from instance masks and verified on contact sheets; the verified label file is shipped with the repository so the benchmark is frozen.

## Repository layout

```
pilot/                 model-agnostic experiment code
  pool.py              geometry filters over COCO annotations
  recolor.py           area-dominant colour proposal from instance masks
  catstats.py          per-category colour statistics (for plausible hints)
  v5data.py            build data/manifest_v5.jsonl (the frozen 2x3 design)
  sheets.py            contact sheets for manual label verification
  infer.py             Qwen2.5-VL scorer, 12-way readout, main run
  infer_gen.py         AutoModelForImageTextToText runner (LLaVA-OneVision-7B)
  probe.py             final-state linear probe, grouped CV, bootstrap CIs
  sdt.py               signal detection and psychometric curves on the probe axis
  decompose.py         answer decomposition and hint adoption vs evidence
  baselines.py         output-space baselines from the same softmax
  selective.py         risk-coverage gates
  blank.py             gray-image text-only control
  matched.py           category-matched pairs (same prompt, different image)
  check_presence.py    presence manipulation check
  steer.py             probe-direction steering sweep
  steer_conflict.py    same direction applied under conflict
  analyze_steer_conflict.py  paired summary for the conflict sweep
  steer_half.py        split-half causal check
  patch3.py            layer-wise patching for Qwen-VL
  patch4.py            layer-wise patching for other families (LLaVA)
  task2.py, task3.py   side / count task-form boundaries
  ext_vizwiz.py        external: natural unanswerable questions
  ext_vqa_color.py     external: VQA-v2 colour questions, frozen probe
data/                  frozen inputs (manifest, verified labels, colour stats)
scripts/               download, build and run scripts
results/               result digest with every number reported in the paper
docs/                  project page (GitHub Pages)
paper/                 LaTeX source
```

## Setup

Python ≥ 3.10, one GPU with ≥ 24 GB (all experiments were run on an A10 24 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_coco.sh        # ~1.1 GB, COCO val2017
bash scripts/download_models.sh      # optional: local snapshots (~30 GB)
bash scripts/build_v5.sh             # rebuilds data/manifest_v5.jsonl + images
```

`download_models.sh` stores snapshots under `models/`; pass them with `--model models/qwen25vl3b`, or omit `--model` to let Hugging Face download directly.

## Reproduce

```bash
# main 2x3 run + all analyses (about 30 min on an A10)
bash scripts/run_main.sh

# steering, split-half steering, layer patching
bash scripts/run_causal.sh

# 7B and LLaVA-OneVision replication
bash scripts/run_cross_model.sh

# controls, task forms, external validation
bash scripts/download_vizwiz.sh
bash scripts/download_vqa2.sh
bash scripts/run_controls.sh
```

Every script accepts `MODEL=` / `MODELS=` environment overrides. `pilot/infer.py` resumes from an existing `.jsonl`, so interrupted runs can be restarted safely. All LLaVA-OneVision stages use a 512 px longest image side (`--resize 512`).

Build the paper and figures (Tectonic downloads the TeX bundle on first use):

```bash
python scripts/make_figures.py
cd paper && tectonic -X compile main.tex
```

## Expected results (Qwen2.5-VL-3B)

Behaviour on 261 samples:

| condition | refusal | mean confidence | adopts hint colour | accuracy |
|---|---|---|---|---|
| present · correct hint | 0.004 | 0.943 | — | 0.973 |
| present · conflicting hint | 0.088 | 0.674 | 0.448 | 0.383 |
| present · no hint | 0.011 | 0.836 | — | 0.762 |
| absent · hint a | 0.521 | 0.699 | 0.467 | — |
| absent · hint b | 0.563 | 0.707 | 0.433 | — |
| absent · no hint | 0.904 | 0.799 | — | — |

Evidence detection (AUROC over 522 paired items):

| signal | no hint | conflicting hint | correct hint |
|---|---|---|---|
| final-state probe | 0.995 | 0.998 | 0.997 |
| p(unknown) | 0.993 | 0.834 | 0.969 |
| max probability | 0.615 | 0.463 | 0.916 |
| 12-d logit probe | 0.982 | 0.923 | 0.974 |

Fabrication at 50% coverage under conflict: probe **0.011**, p(unknown) **0.249**, max-probability **0.510**.

Layer-wise patching, recovery of refusal (3B): L24 (67% depth) 22%, L32 (89%) 80%. 7B: L18 (64%) ~0%, L24 (86%) 69%, L26 (93%) 90%. LLaVA-OneVision-7B: L18 (64%) 36%, L24 (86%) 80%, L26 (93%) 96%.

Full numbers, including 7B/LLaVA, controls, task forms and external validation, are in [`results/summary.md`](results/summary.md) and in the paper.

## What this does and does not claim

- The advantage of internal gating is **specific to hint-polluted settings**. On naturally unanswerable VizWiz images the model's own p(unknown) is as good as the frozen probe (0.878 vs 0.827); the probe still beats max-probability by a wide margin.
- The probe's invariance is **rank invariance**, not calibration invariance: on VQA-v2 the probe's mean score moves from +6.93 to +2.10 under conflict even though its ordering is preserved.
- The probe detects **whether evidence is available**, not whether a particular answer is correct. For answer correctness under no hint, max-probability is the better signal.
- Mechanistic results are strongest on the colour task. Side and count tasks are tested as boundary conditions, not as full replications.

## Citation

```bibtex
@misc{yuanchongbit2026evidencegate,
  title  = {The Evidence Gate: How Vision-Language Models Decide to Answer Without Evidence},
  author = {yuanchongbit},
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {https://github.com/yuanchongbit/evidence-gate}
}
```

## License

MIT for the code. COCO images and annotations follow their own licenses; they are downloaded, not redistributed. The 261-image benchmark is rebuilt from COCO by `scripts/build_v5.sh`.
