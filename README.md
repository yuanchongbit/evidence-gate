# The Evidence Gate

**Conflicting hints make vision-language models answer when the evidence is absent, but they dilute and rotate the internal evidence axis rather than erasing it.**

This repository contains the code, the frozen benchmark and the result summaries for the paper *Hints Dilute and Rotate the Evidence Axis of Vision-Language Models* (preprint in preparation).

Project page: <https://yuanchongbit.github.io/evidence-gate/> (English) · [中文](https://yuanchongbit.github.io/evidence-gate/index.zh.html)

## The question

A VLM is asked *"What color is the umbrella?"* when there is no umbrella in the image. It should say **unknown**. It usually does — until a sentence in the prompt says *"Hint: the umbrella is black."* Then it answers *black*.

Is that a failure of **seeing** or of **reporting what was seen**? We separate the two with a 2×3 within-subject design and find a dissociation:

- **Output-space evidence signals collapse under conflict.** The AUROC of p(unknown) for detecting whether the queried object is present drops from **0.993 to 0.834**; max-probability drops from 0.615 to **0.463** (95% CI [0.413, 0.514]), i.e. to chance.
- **The internal evidence stays decodable, but its axis dilutes and rotates.** A linear probe on the final-token hidden state reaches **0.995** without hints; fitted on training scenes' no-hint states and scored on held-out scenes, it still ranks the conflict arms at **0.958** (3B), **0.946** (7B) and **0.935** (LLaVA, far-hue hint), and refitting inside the condition reaches **0.995–0.999**. The probe's present-minus-absent gap, however, shrinks from **15.9 to 7.7** for the plausible hint and **7.3** for the far-hue hint (ratio 0.46 [0.44, 0.49]) while the within-class spread contracts only to **0.62–0.83×**.
- **The full mean-difference vector both shrinks and turns.** Measured in the state space rather than on one projection, a conflicting hint shortens the present-minus-absent vector to **0.05–0.64** of its no-hint length and rotates it **24°–86°** off the no-hint axis across five backbones (the two hinted colours rotate to almost the same direction, cos 0.72–0.97). The projection a fixed readout sees falls further, to **0.004–0.50**. Noise-normalised separation does not fall and per-scene gaps keep most of their length but lose cross-scene coherence (**0.70 → 0.41** in 7B): the evidence is re-coded, not deleted. Ranking survives; scale and alignment do not.
- **Most of the apparent loss of decision gain is the axis, not the criterion.** On the fixed no-hint axis the conflict refusal curve is 4.2× flatter in 3B; on the condition's own axis it is only 22% flatter (24% in 7B, 12% and n.s. in LLaVA). The dilution is the cross-model invariant (probe-gap ratio 0.05–0.64); in Gemma-3-4B the coarse binary decomposition is a criterion shift rather than a sensitivity loss, so the signal-detection profile, unlike the geometry, is not shared.
- **There is a quantitative dilution→rank-sensitivity regularity, but no universal law.** Across 12 model–condition cells, the fixed-readout sensitivity ratio follows `d′_c/d′_0 = 1.42 × (gap_c/gap_0) + 0.003` (R² = 0.918; through-origin slope 1.425, RMSE 0.060), with a per-model gain of 1.25 (3B), 1.46 (7B), 1.40 (LLaVA) and 1.60 (Gemma) that is stable across hints. It reflects a nearly condition-independent within-class spread contraction (0.707 ± 0.065). A behavioural check removes the identity: with the dependent variable computed from outputs alone (the logit of p(unknown)), the ratio follows the same gap ratio over the eight hinted cells (slope 0.97 with intercept 0.12, R² = 0.57; through-origin 1.27), with a model-specific gain of 0.92 (3B), 1.43 (7B), 1.62 (LLaVA) and 1.40 (Gemma). Out of sample, the probe law holds when the per-model gain is estimated inside the model (leave-one-cell-out MAE 0.056 vs 0.179 for a unit slope; scene-bootstrap slope 1.42 [1.37, 1.47]) but the gain does not extrapolate to an unseen model (leave-one-model-out 0.02–0.12). The behavioural fit is descriptive rather than predictive (LOCO MAE 0.100 vs 0.121–0.125 baselines; LOMO 0.02–0.25). The relation still stops at the output head: it does not predict the models' own confidence collapse, which Gemma dissociates and Phi-3.5 exhibits.
- **The override is late.** Pasting the no-hint state back at layer 32/36 recovers **80%** of the lost refusal (3B); at 26/28 it recovers **90%** (7B); LLaVA-OneVision-7B recovers **36% / 80% / 96%** at 64% / 86% / 93% depth. LLaVA uses a different vision encoder and recipe, though it shares the Qwen LLM lineage.
- **The direction is sufficient at the readout.** Adding or subtracting one probe direction, injected immediately before unembedding, changes abstention bidirectionally and dose-monotonically on held-out halves. This is readout-level causal evidence, not an upstream mechanism claim; ablation is confounded with removed energy.
- **A scene-cross-fitted linear gate helps, within a boundary.** At 25% answered coverage under conflicting hints, a probe fitted only on the no-hint pair reduces fabrication from **0.192** to **0.077** (3B, 2.5×), from **0.169** to **0.146** (7B, nominal 1.2×, **not distinguishable from zero**) and from **0.138** to **0.062** (LLaVA, 2.2×), **0.138 → 0.015** (Gemma-3-4B, 9.0×) and **0.562 → 0.285** (Phi-3.5-vision, 2.0×, where the own signal has reversed). Scene bootstrap of the paired difference at 25%: 3B [−0.169,−0.069], 7B [−0.062,+0.008], LLaVA [−0.138,−0.008], Gemma [−0.185,−0.069], Phi [−0.354,−0.192]; only the 7B advantage is not established. Gemma's operating-point gain comes from threshold placement rather than better ranking (main-pair AUROC 0.887 vs 0.896). At 50% coverage the gain is 2.2× in 3B, 2.0× in LLaVA, 2.2× in Gemma and 1.05× in Phi; 7B refuses 72% of conflict items and cannot reach 50%. The gain survives unanswerable prevalences from 5% to 50%. Conformal certification is **not** supportable at 130 calibration scenes for most models: under a consistent answered-set denominator, a family-wise 10% certificate exists only for Gemma (4/20 splits), and we report the minimum bound (0.174 / 0.210 / 0.191 / 0.119 / 0.305) instead of a certificate. The advantage is regime-specific: on naturally unanswerable VizWiz questions p(unknown) matches the probe.

The effect is not a property of one checkpoint: the same depth profile appears on all five models, with the transition at 69–94% relative depth and 80–102% recovery at the deepest probed layer in every one, and the geometry also appears on two backbones outside the Qwen family. Gemma-3-4B-it (Gemma-3 + SigLIP) reproduces the dilution (**0.53**), the rotation (**24°–29°**), the colour-independent direction (**cos 0.97**) and the held-out transfer (**0.964 / 0.970**), but not the output collapse: its max-probability *rises* under conflict (0.627 → 0.701) while its answers are overridden as strongly as in 3B (accuracy 0.567 → 0.241). Phi-3.5-vision (Phi-3 + CLIP) reproduces the override (absent refusal 0.897 → 0.322/0.425, present refusal 0.119 → 0.548) and pushes the geometry further (dilution **0.052/0.149**, rotation **86°/74°**, coherence **0.39 → 0.03**); its own signal collapses to below chance (**0.939 → 0.385/0.458**) and its frozen probe falls to **0.685/0.675** even though a refit reaches **0.998**. The geometry therefore generalises across three language-model families and three vision encoders; the output collapse appears in the Qwen and Phi readouts but not in Gemma, so it is readout-specific rather than family-wide. A frozen gate still halves fabrication in Phi at 25% coverage (0.562 → 0.285). SmolVLM-Instruct has no abstention behaviour at all (refusal 0 in every cell) and serves as a boundary case.

A second stimulus set built on PASCAL VOC 2012 segmentation (180 images, automatic labels, same 3B checkpoint) reproduces the behaviour, the frozen transfer, the geometry and the gate: conflict lowers absent refusal from 0.894 to 0.644, the frozen probe reaches 0.925/0.950, the evidence vector shortens to 0.41/0.52 and turns 35°/33°, and at 25% answered coverage the gate cuts fabrication from 0.367 to 0.144.

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

# fifth backbone, second outside the Qwen family (Phi-3.5-vision, about 8 min)
bash scripts/run_phi.sh

# second stimulus set (PASCAL VOC 2012; needs the VOC2012 devkit under data/)
bash scripts/run_voc.sh

# v2 controls: matched-energy ablation, no-op patching, prompt formats,
# strict presence, resolution, single-estimator layer transfer, SmolVLM
bash scripts/run_controls_v2.sh

# controls, task forms, external validation
bash scripts/download_vizwiz.sh
bash scripts/download_vqa2.sh
bash scripts/run_controls.sh
```

CPU-only re-analyses of the archived hidden states (no GPU; `.venv` with numpy/scipy/scikit-learn):

```bash
python -m pilot.deploy_check          # cross-fitted gate, matched coverage, conformal re-audit
python -m pilot.geometry_nested --n-boot 500   # nested geometry bootstrap
python -m pilot.validity              # construct and predictive validity
python -m pilot.law_test              # rejected laws + d'-gap relation
python -m pilot.law_behaviour         # behavioural check of the dilution law
python -m pilot.law_validation        # LOCO/LOMO out-of-sample checks + scene bootstrap
python -m pilot.gate_ci               # scene bootstrap CIs for the gate at 25%/50% coverage
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

Fabrication among answered at **matched answered coverage** under conflict, scene-cross-fitted probe / p(unknown) / max-probability (11 colours). At 25% coverage (shared by all models): **0.077 / 0.192 / 0.315** (3B), **0.146 / 0.169 / 0.215** (7B), **0.062 / 0.138 / 0.762** (LLaVA), **0.015 / 0.138 / 0.254** (Gemma) and **0.285 / 0.562 / 0.662** (Phi). At 50% coverage: **0.115 / 0.249 / 0.368** (3B), **0.149 / 0.295 / 0.655** (LLaVA), **0.080 / 0.172 / 0.337** (Gemma) and **0.556 / 0.586 / 0.636** (Phi); 7B answers only 28.4% of conflict items, so 50% is infeasible, and Phi must select 82% of items to answer half. The refit probe reaches 0.100 in 3B at 50% but is an upper bound, not a deployable score. Conformal risk control is underpowered: with a consistent answered-set denominator, a family-wise 10% certificate exists only for Gemma (4/20 splits) and the minimum bounds are 0.174 / 0.210 / 0.191 / 0.119 / 0.305.

Layer-wise patching, recovery of refusal (3B): L24 (67% depth) 22%, L32 (89%) 80%. 7B: L18 (64%) ~0%, L24 (86%) 69%, L26 (93%) 90%. LLaVA-OneVision-7B: L18 (64%) 36%, L24 (86%) 80%, L26 (93%) 96%. Gemma-3-4B: L16 (47%) 12%, L24 (71%) 90%, L30 (88%) 101%. Phi-3.5-vision: L14 (44%) 31%, L22 (69%) 97%, L28 (88%) 102%, L30 (94%) 101%. PASCAL VOC 2012 (3B): L16 (44%) −24%, L24 (67%) 29%, L30 (83%) 84%, L34 (94%) 100%. Same-image present-question states drive refusal below base at the deep layers (−0.70/−0.74 in Gemma, −0.27/−0.33 in Phi), and shuffled absent sources recover as much as the own state in both.

Direction controls (3B / 7B / LLaVA, conflicting pass): ablating `w` lowers absent refusal by 0.215 / 0.211 / 0.102 and raises hint adoption by 0.201 / 0.209 / 0.113, while ablating any of three random directions moves every metric by at most 0.015. This ratio is **not** interpreted as direction-specific necessity: `w` removes 5.4× more state energy than the random directions (projection fraction 0.055 vs 0.019 / 0.004 / 0.008), and class-mean and first-PC directions remove comparable energy (0.052 / 0.066). Matching the removed energy removes most of the effect: `w` scaled to a random direction's energy changes absent refusal by −0.073 (3B), −0.018 (7B) and −0.011 (LLaVA); a random direction scaled up to `w`'s energy is null in 3B (+0.010) but larger than `w` in 7B (−0.044) and LLaVA (−0.029); class-mean and first-PC directions at the same energy move 3B refusal (+0.083), leave 7B unchanged (+0.004) and move LLaVA by −0.063 / −0.038. We therefore claim readout-level sufficiency, not a circuit; the direction is efficient in the Qwen-family models and Phi but not in Gemma. Energy-matched removal in the two non-Qwen models says the same thing. In Gemma the raw ablation is null (0.437 vs base 0.439) and stays null under both matched variants, while class-mean and first-PC directions at `w`'s energy raise refusal by +0.021 each; the head readout depends on class structure, not on `w`. In Phi the raw effect (−0.057) is mostly removed energy: matched down to a random direction's energy it shrinks to −0.013, a random direction scaled up to `w`'s energy moves refusal more and in the opposite direction (+0.048), and class-mean and first-PC directions at matched energy lower it by 0.157 and 0.249. Injection efficiency and removal relevance dissociate. Dose-matched injection is inert through L16 (3B) and L20 (7B) and only bites at L32 / L26; LLaVA's onset is earlier (L20) but still in the last 30% of depth. Same-image, opposite-question patching at L24/L26 gives paired absent-minus-present recovery gaps of +0.51/+0.83 (3B), +0.84/+0.86 (7B), +0.74/+0.80 (LLaVA), +0.71/+0.77 (Gemma) and +0.71/+0.77 (Phi), while shuffled sources restore at least as much. Readout-level steering replicates in Gemma and Phi with different amplitudes and different direction specificity: at ±0.50·‖h‖ absent refusal moves +0.056/−0.109 (Gemma, whose present arm sits at the 0.008 floor) and +0.435/−0.330 (Phi), where conflict leaves the present arm refusing 0.548, so the gate is less specific to the absent question. A random-direction control at the head qualifies this: the fitted direction is 3.0× the mean random effect in Phi (12.9× for ablation) but 0.55× in Gemma, where random directions move the gate more and ablating `w` is null (−0.002) despite removing 5.9× the energy of a random projection. Layer injection is clean only at L30 in Gemma (absent refusal −0.151/+0.088 at ±0.25 with present gold and hint unchanged; L16 and L24 at the negative sign move the gate but wreck the present arm), whereas in Phi the direction already acts at L14 (−0.193/+0.180) and weakens deeper even though patching recovers only 31% at L14 and 97% at L22. We therefore read Gemma's head as movable but non-direction-specific, and Phi's early injection as availability of the axis before the conflict effect is localised, not as its computation site.

Full numbers, including 7B/LLaVA, controls, task forms and external validation, are in [`results/summary.md`](results/summary.md) and in the paper.

## What this does and does not claim

- The advantage of internal gating is **specific to hint-polluted settings**. On naturally unanswerable VizWiz images the model's own p(unknown) is as good as the frozen probe (0.878 vs 0.827); the probe still beats max-probability by a wide margin.
- A **category-matched control** holds the queried category and the asserted colour fixed within a pair, so neither can predict the label. The output collapse, the in-condition refit and the geometry survive, but the frozen probe's margin over p(unknown) falls from **+0.123 / +0.142 / +0.142** (3B / 7B / LLaVA) on the main design to **+0.050 / +0.002 / −0.015**, and its fabrication advantage at 25% coverage shrinks from 0.115 to 0.069 in 3B, disappears in 7B (0.154 vs 0.154) and reverses in LLaVA (0.146 vs 0.054). Part of the main design's frozen transfer is therefore carried by the two arms asking different questions.
- The probe's invariance is **rank invariance**, not calibration invariance: on VQA-v2 the probe's mean score moves from +6.93 to +2.10 under conflict even though its ordering is preserved.
- The probe detects **whether evidence is available**, not whether a particular answer is correct. For answer correctness under no hint, max-probability is the better signal.
- The evidence direction is **not a unique knob at the readout**. Large random directions can also open the gate when added, at matched norm, and removing `w` is confounded with removing more state energy (5.4× the random directions). We therefore do not claim direction-specific necessity. What the intervention shows is sufficiency at the readout: adding the direction changes abstention bidirectionally, so the gate is expressed late and the evidence axis remains connected to it.
- The absent arm carries a **scene-fit component**: absent categories that co-occur more with the image's own objects are refused less (by 0.116 / 0.069 / 0.132 in 3B / 7B / LLaVA). Category frequency is only weakly related to scene fit (Spearman 0.23) and does not explain it: in a joint fit scene fit stays negative in all three models and frequency adds only in 7B. Evidence detection survives within both strata (p(unknown) AUROC ≥ 0.976, probe ≥ 0.989), so scene fit moves the level of the gate, not the ranking of evidence.
- SmolVLM-Instruct (a non-Qwen backbone) is a **boundary case, not a replication**: it never emits `unknown` in any of the six cells (refusal 0.000; adoption 0.985 / 0.989), so there is no gate to open or close. Its final state still separates present from absent images (probe 0.989 with no hint, 0.993 under conflict), with the caveat that on conflict pairs the probe may be reading hint–image mismatch rather than evidence. Phi-3.5-vision, by contrast, does abstain (0.897) and is the model in which the evidence axis rotates furthest (86°) while a refit still decodes it at 0.998.
- Mechanistic results are strongest on the colour task. Side and count tasks are tested as boundary conditions, not as full replications.

## Citation

```bibtex
@misc{yuanchongbit2026evidencegate,
  title  = {Hints Dilute and Rotate the Evidence Axis of Vision-Language Models},
  author = {Chong Yuan},
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {https://github.com/yuanchongbit/evidence-gate}
}
```

## License

MIT for the code. COCO images and annotations follow their own licenses; they are downloaded, not redistributed. The 261-image benchmark is rebuilt from COCO by `scripts/build_v5.sh`.
