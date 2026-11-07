# The Evidence Gate

**Conflicting hints make vision-language models answer when the evidence is absent, but they dilute and rotate the internal evidence axis rather than erasing it.**

This repository contains the code, the frozen benchmark and the result summaries for the paper *Hints Dilute and Rotate the Evidence Axis of Vision-Language Models* (preprint in preparation).

Project page: <https://yuanchongbit.github.io/evidence-gate/> (English) · [中文](https://yuanchongbit.github.io/evidence-gate/index.zh.html)

## The question

A VLM is asked *"What color is the umbrella?"* when there is no umbrella in the image. It should say **unknown**. It usually does — until a sentence in the prompt says *"Hint: the umbrella is black."* Then it answers *black*.

Is that a failure of **seeing** or of **reporting what was seen**? We separate the two with a 2×3 within-subject design and find a dissociation:

- **Output-space evidence signals collapse under conflict.** The AUROC of p(unknown) for detecting whether the queried object is present drops from **0.993 to 0.834**; max-probability drops from 0.615 to **0.463** (95% CI [0.413, 0.514]), i.e. to chance.
- **The internal evidence stays decodable, but its axis dilutes and rotates.** A linear probe on the final-token hidden state reaches **0.995** without hints; fitted on training scenes' no-hint states and scored on held-out scenes, it still ranks the conflict arms at **0.958** (3B), **0.946** (7B) and **0.935** (LLaVA, far-hue hint), and refitting inside the condition reaches **0.995–0.999**. The probe's present-minus-absent gap, however, shrinks from **15.9 to 7.7** for the plausible hint and **7.3** for the far-hue hint (ratio 0.46 [0.44, 0.49]); the within-class spread is unchanged.
- **The full mean-difference vector both shrinks and turns.** Measured in the state space rather than on one projection, a conflicting hint shortens the present-minus-absent vector to **0.39–0.64** of its no-hint length and rotates it **39°–71°** off the no-hint axis (both hinted colours rotate to almost the same direction, cos 0.91). The projection a fixed readout sees falls further, to **0.20–0.43**. Noise-normalised separation does not fall and per-scene gaps keep most of their length but lose cross-scene coherence (**0.70 → 0.41** in 7B): the evidence is re-coded, not deleted. Ranking survives; scale and alignment do not.
- **Most of the apparent loss of decision gain is the axis, not the criterion.** On the fixed no-hint axis the conflict refusal curve is 4.2× flatter in 3B; on the condition's own axis it is only 22% flatter (24% in 7B, 12% and n.s. in LLaVA). The dilution is the cross-model invariant (probe-gap ratio 0.34–0.49); in Gemma-3-4B the coarse binary decomposition is a criterion shift rather than a sensitivity loss, so the signal-detection profile, unlike the geometry, is not shared.
- **The override is late.** Pasting the no-hint state back at layer 32/36 recovers **80%** of the lost refusal (3B); at 26/28 it recovers **90%** (7B); LLaVA-OneVision-7B recovers **36% / 80% / 96%** at 64% / 86% / 93% depth. LLaVA uses a different vision encoder and recipe, though it shares the Qwen LLM lineage.
- **The direction is sufficient at the readout.** Adding or subtracting one probe direction, injected immediately before unembedding, changes abstention bidirectionally and dose-monotonically on held-out halves. This is readout-level causal evidence, not an upstream mechanism claim; ablation is confounded with removed energy.
- **A frozen linear gate works where it matters, within a boundary.** At 50% coverage under conflicting hints, ranking by the frozen probe reduces fabrication from **0.249** to **0.069** (3B, 3.6×), from **0.268** to **0.126** (7B, 2.1×) and from **0.295** to **0.111** (LLaVA, 2.7×). The gain survives unanswerable prevalences from 5% to 50%, and a 10% conformal risk certificate holds in two of three models; a 5% certificate is infeasible at 130 calibration scenes. The advantage is regime-specific: on naturally unanswerable VizWiz questions p(unknown) matches the probe.

The effect is not a property of one checkpoint: the same depth profile appears on Qwen2.5-VL-3B, Qwen2.5-VL-7B and LLaVA-OneVision-7B, and the geometry also appears on Gemma-3-4B-it (Gemma-3 + SigLIP, the only backbone whose language model is not from the Qwen family). Gemma reproduces the dilution (**0.53**), the rotation (**24°–29°**), the colour-independent direction (**cos 0.97**) and the held-out transfer (**0.964 / 0.970**), but not the output collapse: its max-probability *rises* under conflict (0.627 → 0.701) while its answers are overridden as strongly as in 3B (accuracy 0.567 → 0.241). The representational result therefore generalises across language-model families; the rank–scale dissociation is specific to the Qwen2/2.5 readouts in this battery. SmolVLM-Instruct has no abstention behaviour at all (refusal 0 in every cell) and serves as a boundary case.

## Design

|  | no hint | correct hint | conflicting hint |
|---|---|---|---|
| **object present** | `sufficient\|none` | `sufficient\|correct` | `sufficient\|wrong` |
| **object annotation-absent** | `absent\|none` | `absent\|a` | `absent\|b` |

Every image appears in all six cells. The absent arm asks about an object category that COCO's exhaustive val2017 annotations place outside the image; the hinted colour is that category's most frequent colour in COCO val, so the hint is plausible rather than adversarial. **No pixel is edited.** A separate yes/no presence question confirms the manipulation (p(yes) = 0.046 on the main absent arm and 0.095 on 260 category-matched negatives, vs. 0.95 for the original images).

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
  geometry.py          full mean-difference geometry: dilution, rotation, coherence, held-out transfer
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
  control_steer.py     random-direction and dose-matched layer controls
  patch3.py            layer-wise patching for Qwen-VL
  patch4.py            layer-wise patching for other families (LLaVA)
  patch5.py            same-image opposite-question patching control
  control_analyze.py   summaries for the causal-control runs
  extra44.py           energy-matched ablations, direction competitors, per-prompt dosing
  patch6.py            no-op and zero-paste patching controls
  prompt_variants.py   no-unknown and sentence-hint prompt formats
  presence_main.py     yes/no presence check on the main absent arm
  resolution3b.py      resolution-matched rerun of the 3B battery
  layer_transfer.py    single-estimator per-layer transfer
  plausibility.py      scene-fit stratification of the absent arm
  analyze44.py         summaries for the v2 control battery
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

# random-direction, dose-matched layer and same-image patching controls
bash scripts/run_control_causal.sh

# 7B and LLaVA-OneVision replication
bash scripts/run_cross_model.sh

# fourth backbone outside the Qwen family (Gemma-3-4B-it, about 6 min)
bash scripts/run_nonqwen.sh

# v2 controls: matched-energy ablation, no-op patching, prompt formats,
# strict presence, resolution, single-estimator layer transfer, SmolVLM
bash scripts/run_controls_v2.sh

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
| frozen final-state probe (scene-held-out) | 0.995 | 0.958 | 0.990 |
| refit final-state probe (upper bound) | 0.995 | 0.995 | 1.000 |
| p(unknown) | 0.993 | 0.834 | 0.969 |
| max probability | 0.615 | 0.463 | 0.916 |
| 12-d logit probe | 0.982 | 0.923 | 0.974 |

The frozen probe is fitted on no-hint scenes and scored on held-out scenes (C=0.1); scoring it on its own fitting scenes gives 0.971, so scene overlap inflates the estimate by 0.013 here (at most 0.035 across models and hints), and a C grid moves the held-out value by at most 0.020. Full geometry (vector length, rotation, Mahalanobis length, per-scene coherence) is in `results/geometry.json` and Section 4.3 of the paper.

Fabrication at 50% coverage under conflict, frozen probe / p(unknown) / max-probability: **0.069 / 0.249 / 0.510** (3B), **0.126 / 0.268 / 0.648** (7B), **0.111 / 0.295 / 0.655** (LLaVA). The refit probe reaches 0.011 in 3B but is an upper bound, not a deployable score.

Layer-wise patching, recovery of refusal (3B): L24 (67% depth) 22%, L32 (89%) 80%. 7B: L18 (64%) ~0%, L24 (86%) 69%, L26 (93%) 90%. LLaVA-OneVision-7B: L18 (64%) 36%, L24 (86%) 80%, L26 (93%) 96%.

Direction controls (3B / 7B / LLaVA, conflicting pass): ablating `w` lowers absent refusal by 0.215 / 0.211 / 0.102 and raises hint adoption by 0.201 / 0.209 / 0.113, while ablating any of three random directions moves every metric by at most 0.015. This ratio is **not** interpreted as direction-specific necessity: `w` removes 5.4× more state energy than the random directions (projection fraction 0.055 vs 0.019 / 0.004 / 0.008), and class-mean and first-PC directions remove comparable energy (0.052 / 0.066). Matching the removed energy removes most of the effect: `w` scaled to a random direction's energy changes absent refusal by −0.073 (3B), −0.018 (7B) and −0.011 (LLaVA); a random direction scaled up to `w`'s energy is null in 3B (+0.010) but larger than `w` in 7B (−0.044) and LLaVA (−0.029); class-mean and first-PC directions at the same energy move 3B refusal (+0.083), leave 7B unchanged (+0.004) and move LLaVA by −0.063 / −0.038. We therefore claim readout-level sufficiency and an efficient readout direction, not a circuit. Dose-matched injection is inert through L16 (3B) and L20 (7B) and only bites at L32 / L26; LLaVA's onset is earlier (L20) but still in the last 30% of depth. Same-image, opposite-question patching at L24/L26 gives paired absent-minus-present recovery gaps of +0.51/+0.83 (3B), +0.84/+0.86 (7B) and +0.74/+0.80 (LLaVA), while shuffled sources restore at least as much.

Full numbers, including 7B/LLaVA, controls, task forms and external validation, are in [`results/summary.md`](results/summary.md) and in the paper.

## What this does and does not claim

- The advantage of internal gating is **specific to hint-polluted settings**. On naturally unanswerable VizWiz images the model's own p(unknown) is as good as the frozen probe (0.878 vs 0.827); the probe still beats max-probability by a wide margin.
- The probe's invariance is **rank invariance**, not calibration invariance: on VQA-v2 the probe's mean score moves from +6.93 to +2.10 under conflict even though its ordering is preserved.
- The probe detects **whether evidence is available**, not whether a particular answer is correct. For answer correctness under no hint, max-probability is the better signal.
- The evidence direction is **not a unique knob at the readout**. Large random directions can also open the gate when added, at matched norm, and removing `w` is confounded with removing more state energy (5.4× the random directions). We therefore do not claim direction-specific necessity. What the intervention shows is sufficiency at the readout: adding the direction changes abstention bidirectionally, so the gate is expressed late and the evidence axis remains connected to it.
- The absent arm carries a **scene-fit component**: absent categories that co-occur more with the image's own objects are refused less (by 0.116 / 0.069 / 0.132 in 3B / 7B / LLaVA). Category frequency is only weakly related to scene fit (Spearman 0.23) and does not explain it: in a joint fit scene fit stays negative in all three models and frequency adds only in 7B. Evidence detection survives within both strata (p(unknown) AUROC ≥ 0.976, probe ≥ 0.989), so scene fit moves the level of the gate, not the ranking of evidence.
- SmolVLM-Instruct (a non-Qwen backbone) is a **boundary case, not a fourth replication**: it never emits `unknown` in any of the six cells (refusal 0.000; adoption 0.985 / 0.989), so there is no gate to open or close. Its final state still separates present from absent images (probe 0.989 with no hint, 0.993 under conflict), with the caveat that on conflict pairs the probe may be reading hint–image mismatch rather than evidence.
- Mechanistic results are strongest on the colour task. Side and count tasks are tested as boundary conditions, not as full replications.

## Citation

```bibtex
@misc{yuanchongbit2026evidencegate,
  title  = {Hints Dilute and Rotate the Evidence Axis of Vision-Language Models},
  author = {yuanchongbit},
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {https://github.com/yuanchongbit/evidence-gate}
}
```

## License

MIT for the code. COCO images and annotations follow their own licenses; they are downloaded, not redistributed. The 261-image benchmark is rebuilt from COCO by `scripts/build_v5.sh`.
