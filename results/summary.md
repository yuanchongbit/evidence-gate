# Result digests

These are the numbers reported in the paper, transcribed from the stdout of the
analysis scripts on a single 24 GB A10. The scripts are deterministic given the
frozen inputs (fixed seeds, grouped CV, bootstrap with fixed resamples); rerun
them to regenerate the underlying `.jsonl` / `.npz` files and full logs.

| Run | Script | Raw stdout (remote) |
|---|---|---|
| Main 2x3 behaviour | `pilot/analyze_v5.py` | `runs/repro/main_3b.txt` |
| Evidence probe | `pilot/probe.py` | `runs/repro/probe_3b.txt` |
| Signal detection | `pilot/sdt.py` | `results/sdt.json` |
| Answer decomposition / hint adoption | `pilot/decompose.py` | `results/decompose*.json` |
| Output-space baselines | `pilot/baselines.py` | `runs/repro/baselines_3b.txt` |
| Selective gate | `pilot/selective.py` | `runs/repro/selective_3b.txt` |
| Steering / held-out half | `pilot/steer.py`, `pilot/steer_half.py` | `runs/repro/steer_3b.txt`, `steer_half.txt` |
| Steering under conflict (3B / 7B / LLaVA) | `pilot/steer_conflict.py`, `pilot/analyze_steer_conflict.py` | `results/steer_conflict{,_7b,_llava}.json` |
| Layer patching | `pilot/patch3.py`, `pilot/patch4.py` | `runs/repro/patch*.txt` |
| Task forms | `pilot/task2.py`, `pilot/task3.py` | `runs/repro/task2.txt`, `task3.txt` |
| External validation | `pilot/ext_analyze.py`, `pilot/ext_vqa_analyze.py` | `runs/ext_*_summary.json` |
| Deployment re-audit (cross-fit, matched coverage) | `pilot/deploy_check.py` | `results/deploy_check.json` |
| Nested geometry bootstrap | `pilot/geometry_nested.py` | `results/geometry_nested.json` |
| Construct / predictive validity | `pilot/validity.py` | `results/validity.json` |
| Law tests (isotropic contraction, d'-gap fit) | `pilot/law_test.py` | `results/law_isotropic.json` |
| Behavioural check of the dilution law | `pilot/law_behaviour.py` | `results/law_behaviour.json` |
| Category-matched control (LLaVA) | `pilot/matched_gen.py` | `results/matched_llava.jsonl`, `.npz` |
| Matched analysis / gate (3B / 7B / LLaVA) | `pilot/matched_analysis.py`, `pilot/matched_gate.py` | `results/matched.json`, `results/matched_gate.json` |
| Second stimulus set (PASCAL VOC 2012, 3B) | `pilot.voc_analyze`, `pilot.voc_extra` | `results/voc_results.json`, `results/voc_extra.json` |
| Second non-Qwen abstaining backbone (Phi-3.5-vision) | `pilot.voc_analyze`, `pilot.voc_extra` | `results/phi_results.json`, `results/phi_extra.json` |

The paper's controls (gray images, category-matched pairs, category-grouped CV)
come from `pilot/blank.py`, `pilot/matched.py` and `pilot/matched_gen.py`; their
key numbers are repeated below.

## 1. Behaviour (Qwen2.5-VL-3B, n=261)

| condition | n | refusal | mean conf. | adopts hint | accuracy |
|---|---|---|---|---|---|
| present, correct hint | 261 | 0.004 | 0.943 | 0.000 | 0.973 |
| present, conflicting hint | 261 | 0.088 | 0.674 | 0.448 | 0.383 |
| present, no hint | 261 | 0.011 | 0.836 | — | 0.762 |
| absent, hint a | 261 | 0.521 | 0.699 | 0.467 | — |
| absent, hint b | 261 | 0.563 | 0.707 | 0.433 | — |
| absent, no hint | 261 | 0.904 | 0.799 | — | — |

UAR (answers when the object is absent) 0.337; OAR (refuses when present) 0.034.
Presence check on 260 category-matched negatives: mean p(yes) = 0.095, versus
0.95 on the original images.

## 2. Evidence detection, 3B (AUROC over 522 paired items)

| signal | no hint | conflicting | correct |
|---|---|---|---|
| frozen final-state probe (headline) | 0.995 [0.989, 0.999] | 0.957 [0.939, 0.974] | 0.990 [0.981, 0.997] |
| refit final-state probe (upper bound) | 0.995 [0.989, 0.999] | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] |
| p(unknown) | 0.993 | 0.834 | 0.969 |
| max probability | 0.615 | 0.463 | 0.916 |
| margin (colour max - unknown) | 0.985 | 0.766 | 0.959 |
| negative entropy | 0.640 | 0.364 | 0.902 |
| top-2 gap | 0.934 | 0.564 | 0.927 |
| 12-d logit probe | 0.982 | 0.923 | 0.974 |

Probe minus p(unknown): no hint +0.002 [-0.003, +0.007]; conflict +0.123
[+0.098, +0.148]; correct +0.028 [+0.017, +0.042].

Transfer, fitted on the no-hint pair and evaluated on the hinted pair, with
scenes held out (5 scene folds: fit on training scenes' no-hint states, score
held-out scenes' hinted states) and C fixed at 0.1. The frozen row above is this
held-out transfer; refitting inside the condition is an upper bound.

| model | pair | leaky (fit-scene scores) | held out | refit |
|---|---|---|---|---|
| 3B | conflict a | 0.969 | 0.957 [0.939, 0.974] | 0.997 |
| 3B | conflict b | 0.971 | 0.958 [0.939, 0.974] | 0.995 |
| 3B | correct | 0.996 | 0.990 [0.981, 0.997] | 0.997 |
| 7B | conflict a | 0.953 | 0.939 [0.919, 0.956] | 0.997 |
| 7B | conflict b | 0.955 | 0.946 [0.926, 0.963] | 0.993 |
| 7B | correct | 0.999 | 0.997 [0.995, 0.999] | 0.998 |
| LLaVA | conflict a | 0.958 | 0.923 [0.899, 0.945] | 0.999 |
| LLaVA | conflict b | 0.960 | 0.935 [0.914, 0.954] | 0.998 |
| LLaVA | correct | 0.997 | 0.986 [0.978, 0.993] | 0.998 |

Scene overlap inflates the number, though not much: leaky minus held-out is
0.002-0.035 (largest for LLaVA, hint a). A C grid (0.0003, 0.001, 0.003, 0.01,
0.1) under the held-out protocol moves conflict AUROC by at most 0.020 (LLaVA,
hint a). Reverse
transfer (conflict -> none) is 0.999. Headline numbers in the paper are held out
at C=0.1.

## 2b. Signal detection / psychometric curves, 3B

The probe score (GroupKFold by scene, applied out of fold; C=0.1, the main
probe's regularisation) is the evidence axis. Two fits are run. The *fixed*
axis takes z from the no-hint forward pass of each item, so the hint cannot
move it. The *own* axis takes z from the condition's own forward pass, projected
on the same no-hint direction, so only the state changes. 261 scenes, 1,827
item-condition observations. Binary SDT uses empirical refusal rates on the
paired items. Separation is the present-minus-absent gap of the fixed no-hint
probe in pooled within-arm SD units.

| condition | d' | criterion c | separation | slope fixed | slope own |
|---|---|---|---|---|---|
| no hint | 3.51 | 0.46 | 5.06 | -0.719 | -0.719 |
| conflicting hint | 1.45 | 0.62 | 2.60 | -0.171 | -0.559 |
| correct hint | 3.83 | 0.62 | 4.17 | -0.680 | -0.641 |

Predicted probabilities are at the median no-hint score of absent (z=-8.56)
and present (z=7.43) items. Cluster bootstrap over scenes, 2,000 resamples.
Paired changes vs no hint:

- conflict: delta d' = -2.13 [-2.69, -1.66]; delta c = +0.16 [-0.18, +0.39]
- correct: delta d' = +0.31 [-0.36, +0.93]; delta c = +0.16 [-0.18, +0.46]

Conflict raises refusal on present items (0.011 -> 0.088) and lowers it on
absent items (0.904 -> 0.542). Fixed-axis fit: b0=-1.780, a=-0.719,
b_conflict=+0.629, a_conflict=+0.548, b_correct=+0.007, a_correct=+0.039.
Slope change: +0.548 [0.439, 0.774]; intercept change: +0.629 [0.039, 1.569].
Non-parallel likelihood-ratio 178.2, df=2, p=2e-39. Correct hint: slope change
+0.039 [-0.074, 0.196] (n.s. for both parameters). The conflict curves cross
near z=-1.2, so this is a pivot (compression), not a translation.

Own-axis fit: b_conflict=-0.561 [-1.282, +0.448], a_conflict=+0.160
[+0.013, +0.429]; LR 6.4, p=0.041. In its own coordinates the decision is 22%
flatter, not four times flatter. The geometry explains why: under conflict the
fixed no-hint direction's present-minus-absent gap contracts from 15.86 to
7.69 for hint a (ratio 0.485 [0.460, 0.511]) and to 7.34 for hint b (0.463
[0.438, 0.488]), while within-arm SDs stay close
(2.83/3.09 -> 2.56/2.88). Applying the own-axis slope to the contracted axis
predicts a 2.7-fold flattening; the remainder is item-level variation in the
contraction (r=0.855 between the pre- and post-hint projections). The fixed
direction still ranks conflict pairs at 0.957 held out (0.958 for the far-hue
hint); a direction refit on the conflict pair reaches 0.997. Reading: the hint
dilutes and rotates the evidence axis; ranking survives, scale does not. The
residual 22% gain loss is real but modest. The criterion shift is 3B-specific
and inconsistent across models.
Figure: paper/figures/fig_sdt.pdf.

## 2c. Evidence-axis geometry: dilution and rotation

`pilot/geometry.py` -> `results/geometry.json`. For each condition it computes
the full mean-difference vector `dmu = mean(h_present) - mean(h_absent)`: its
norm relative to the no-hint norm, its angle to the no-hint unit axis `w`, its
projection on `w`, its Mahalanobis length under no-hint within-class whitening
(ridge 1% of mean eigenvalue), and the per-scene paired-gap coherence
`||mean_s v_s|| / rms_s ||v_s||`. Scene bootstrap, 500 draws. The same
estimator applied to the no-hint vector against itself gives ratio 1.00
[0.97, 1.05] and angle 3-5 deg: the noise floor.

| model | cond | \|dmu\| ratio | angle to w | along-w ratio | Mahalanobis (none -> cond) | coherence (none -> cond) |
|---|---|---|---|---|---|---|
| 3B | conflict a | 0.528 [0.502, 0.554] | 39.2 [35.9, 42.5] | 0.409 | 0.450 -> 0.518 | 0.597 -> 0.397 |
| 3B | conflict b | 0.572 [0.544, 0.597] | 40.7 [37.7, 43.9] | 0.433 | 0.450 -> 0.491 | 0.597 -> 0.444 |
| 3B | correct | 0.780 [0.749, 0.817] | 39.1 [37.4, 41.2] | 0.605 | 0.450 -> 0.682 | 0.597 -> 0.541 |
| 7B | conflict a | 0.387 [0.362, 0.413] | 47.7 [44.3, 51.4] | 0.259 | 0.428 -> 0.548 | 0.704 -> 0.409 |
| 7B | conflict b | 0.430 [0.402, 0.456] | 44.3 [41.8, 47.2] | 0.307 | 0.428 -> 0.530 | 0.704 -> 0.465 |
| 7B | correct | 0.746 [0.717, 0.779] | 22.4 [21.4, 23.7] | 0.689 | 0.428 -> 0.716 | 0.704 -> 0.641 |
| LLaVA | conflict a | 0.606 [0.572, 0.640] | 70.8 [68.1, 73.2] | 0.199 | 0.424 -> 0.634 | 0.605 -> 0.448 |
| LLaVA | conflict b | 0.644 [0.607, 0.679] | 68.7 [66.1, 70.9] | 0.234 | 0.424 -> 0.619 | 0.605 -> 0.485 |
| LLaVA | correct | 0.687 [0.654, 0.723] | 45.5 [43.3, 47.9] | 0.482 | 0.424 -> 0.598 | 0.605 -> 0.482 |

Directions of the two hinted colours agree: cos(conflict a, conflict b) =
0.908 (3B), 0.908 (7B), 0.915 (LLaVA); cos(none, conflict) is 0.783 / 0.676 /
0.331. Per-scene gap norms fall less than the mean vector (3B 78.5 -> 61.7, 7B
155.4 -> 102.9, LLaVA 108.4 -> 87.9), so part of the aggregate loss is
decorrelation across scenes rather than uniform shrinkage. Noise-normalised
separation does not fall, and an in-condition refit reaches 0.995-0.999: the
evidence is re-coded, not removed. Reading: a hint dilutes the full vector and
rotates it systematically off the direction a no-hint readout uses; per-scene
gaps also stop agreeing. The projection on `w` falls further than either factor
alone because it is their product.

Nested bootstrap (`pilot/geometry_nested.py`, 500 scene resamples; `W`, `w`,
`d0` re-estimated inside every resample) reproduces the same picture with
intervals that exclude the no-effect value: conflict ratio / angle / along-w
are 0.523 [0.508, 0.544] / 38.5 deg [35.7, 42.1] / 0.409 [0.381, 0.438] in
3B; 0.384 [0.362, 0.409] / 47.4 [44.1, 51.1] / 0.259 [0.230, 0.291] in 7B;
0.602 [0.572, 0.637] / 70.7 [67.9, 73.1] / 0.199 [0.175, 0.227] in LLaVA;
0.529 [0.499, 0.561] / 28.2 [26.6, 30.7] / 0.466 [0.435, 0.496] in Gemma.
cos(a, b) is 0.908 [0.877, 0.929], 0.908 [0.868, 0.931], 0.915 [0.884, 0.933]
and 0.971 [0.958, 0.977]. Every interval for angle excludes zero and every
ratio interval excludes one, so dilution and rotation are not artefacts of
estimating the axis once on all scenes.

Cross-model binary SDT (same 261 scenes, 522 absent items under conflict):

| model | d' no hint | d' conflict | delta d' [95% CI] | c no hint | c conflict | delta c [95% CI] |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 3.51 | 1.45 | -2.13 [-2.69, -1.66] | 0.46 | 0.62 | +0.16 [-0.18, +0.39] |
| Qwen2.5-VL-7B | 2.98 | 1.24 | -1.80 [-2.14, -1.39] | -0.22 | -0.76 | -0.54 [-0.73, -0.35] |
| LLaVA-OneVision-7B | 3.44 | 1.44 | -2.06 [-2.71, -1.32] | 0.81 | 1.81 | +1.00 [+0.66, +1.33] |

Cross-model axis compression and own-axis gain change:

| model | gap ratio [95% CI] | delta slope, fixed | delta slope, own [95% CI] | LR fixed | LR own |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 0.463 [0.438, 0.488] (b) | +0.548 | +0.160 [+0.013, +0.429] | 178.2 | 6.4 (p=0.041) |
| Qwen2.5-VL-7B | 0.424 [0.393, 0.457] (b) | +0.332 | +0.113 [+0.025, +0.232] | 164.8 | 22.0 (p=2e-5) |
| LLaVA-OneVision-7B | 0.344 [0.324, 0.365] (b) | +0.286 | +0.053 [-0.057, +0.185] | 40.8 | 1.6 (n.s.) |

The sensitivity collapse replicates in all three; the criterion shift changes
sign across models (null / negative / positive), so a uniform threshold account
is ruled out. The contraction is the cross-model invariant; the residual
own-axis gain loss is 12-24% and is not distinguishable from zero on LLaVA.
Under a correct hint d' never falls: 3B +0.31 [-0.36, +0.93],
7B +0.50 [+0.17, +0.82], LLaVA +0.32 [0.00, +0.68].

## 2c. Failure modes and hint adoption (all models)

Argmax over the 12-way readout; `hint` is the colour named in the prompt.
For present items the hint is the distractor colour; for absent items hint a
is the named colour and hint b is its mapped distractor. `other` is any other
colour. Rows sum to 1 within rounding.

| model | condition | unknown | true colour | hint | other |
|---|---|---|---|---|---|
| 3B | present, conflict | 0.088 | 0.383 | 0.448 | 0.080 |
| 3B | absent, hint a | 0.521 | — | 0.467 | 0.011 |
| 3B | absent, hint b | 0.563 | — | 0.433 | 0.004 |
| 7B | present, conflict | 0.556 | 0.249 | 0.176 | 0.019 |
| 7B | absent, hint a | 0.877 | — | 0.123 | 0.000 |
| 7B | absent, hint b | 0.958 | — | 0.042 | 0.000 |
| LLaVA | present, conflict | 0.004 | 0.360 | 0.490 | 0.146 |
| LLaVA | absent, hint a | 0.088 | — | 0.904 | 0.008 |
| LLaVA | absent, hint b | 0.188 | — | 0.762 | 0.050 |

The three models fail differently: hint takeover (3B), over-refusal (7B),
near-total credulity (LLaVA). The sensitivity loss is what they share.

Visual evidence still grades the answer. On the 261 present-conflict items,
regressing the outcome on the no-hint probe score (logit, cluster bootstrap
over scenes, 2,000 resamples):

- P(argmax = hint) ~ z: slope -0.213 [-0.325, -0.119]
- P(argmax = gold) ~ z: slope +0.267 [0.163, 0.401]

So the hint tilts the decision, but a higher pre-hint evidence score makes
hint adoption less likely and the true colour more likely.

## 2d. Construct and predictive validity

`pilot/validity.py` -> `results/validity.json`. Construct check: the no-hint
probe separates present from absent items even when the present arm is
restricted to items the model answered *wrong*, so the separation is not
carried by correctness on the present arm. Within-present correct-vs-wrong is
much weaker, so the probe tracks presence, not answerability.

| model | present vs absent, all | present answered-correct | present answered-wrong | within present, correct vs wrong |
|---|---|---|---|---|
| 3B | 0.995 [0.989, 0.999] | 0.996 [0.989, 1.000] | 0.991 [0.980, 1.000] | 0.619 [0.538, 0.697] |
| 7B | 0.996 [0.993, 0.999] | 0.998 [0.995, 1.000] | 0.991 [0.981, 0.999] | 0.572 [0.492, 0.652] |
| LLaVA | 0.994 [0.989, 0.998] | 0.995 [0.988, 0.999] | 0.994 [0.985, 1.000] | 0.602 [0.532, 0.671] |
| Gemma | 0.985 [0.974, 0.994] | 0.985 [0.969, 0.996] | 0.985 [0.969, 0.995] | 0.586 [0.515, 0.656] |

Predictive check: out-of-fold conflict score (fitted on no-hint, scored on
held-out scenes) against three conflict behaviours. AUROC [95% scene
bootstrap]; `p` is the Spearman permutation test.

| model | absent refuse | present gold | present adopt |
|---|---|---|---|
| 3B | 0.644 [0.581, 0.708] | 0.549 [0.478, 0.623] | 0.448 [0.379, 0.520] |
| 7B | 0.695 [0.587, 0.794] | 0.567 [0.481, 0.653] | 0.503 [0.405, 0.601] |
| LLaVA | 0.454 [0.346, 0.566] | 0.529 [0.450, 0.606] | 0.477 [0.404, 0.552] |
| Gemma | 0.750 [0.689, 0.807] | 0.676 [0.596, 0.753] | 0.334 [0.260, 0.414] |

The probe reliably predicts absent-item refusal in 3B, 7B and Gemma, and in
Gemma it also predicts whether a correct present answer survives (`rho=0.26`)
and whether the hint is adopted (`rho=-0.25`). It does not predict adoption in
3B/7B, and in LLaVA none of the three behaviours is predicted because refusal
is near zero (0.088) and adoption near ceiling (0.636), leaving little
variance. The gate therefore works on average (Section 6) without being a
calibrated per-item predictor of credulity. This is a real scope limit: the
score measures evidence availability, not the strength of the hint override.


## 3. Controls

### 3.1 Category-matched pairs

The main design's present and absent arms query different categories and assert
different colours, so the question and the hint can predict the label. The
control pairs each present image with a different image in which the same
category is annotation-absent; the question and the asserted colour are
identical within a pair, and the asserted colour is never the present image's
gold. 260 pairs over 30 categories (520 items, 246 images; 157 images appear as
both a present image and a negative).

| model | no hint | frozen | refit | p(unknown) | frozen − p(unknown) | geometry (len / rot / along) |
|---|---|---|---|---|---|---|
| 3B | 0.983 | 0.867 | 0.975 | 0.817 | +0.050 [0.028, 0.077] | 0.565 / 39.6° / 0.435 |
| 7B | 0.983 | 0.823 | 0.978 | 0.820 | +0.002 [−0.025, 0.045] | 0.394 / 45.9° / 0.274 |
| LLaVA | 0.994 | 0.855 | 0.983 | 0.870 | −0.015 [−0.062, 0.025] | 0.586 / 65.9° / 0.240 |

AUROC over the 520 matched items; folds grouped by category. `no hint` and
`refit` are out of fold. `frozen` fits the no-hint probe on training categories
and scores the hinted states of held-out categories. Margins are
category-grouped bootstrap 95% CIs. Grouping folds by image instead of category
leaves the frozen and refit values within 0.01 (0.876 / 0.984 in 3B,
0.833 / 0.980 in 7B, 0.869 / 0.989 in LLaVA).

The output collapse, the in-condition decodability and the geometry survive:
p(unknown) falls from 0.983 to 0.817, from 0.975 to 0.820 and from 0.989 to
0.870; maximum-probability AUROC from 0.952 to 0.699, from 0.967 to 0.768 and
from 0.949 to 0.510; refitting recovers 0.975 / 0.978 / 0.983; and the
present−absent mean difference shortens to 0.565 / 0.394 / 0.586 of its
no-hint length and rotates 39.6° / 45.9° / 65.9°, against
0.523 / 0.384 / 0.602 and 38.5° / 47.4° / 70.7° for the plausible hint on the
main pair. What shrinks is frozen transfer: 0.867 [0.828, 0.901],
0.823 [0.776, 0.869] and 0.855 [0.805, 0.900] against 0.957 / 0.939 / 0.923 on
the main pair, and margins over p(unknown) of +0.050 / +0.002 / −0.015 against
+0.123 / +0.142 / +0.142; in 7B and LLaVA the margin is indistinguishable from
zero and in LLaVA the point estimate is negative. Part of the main design's
frozen transfer is therefore carried by the change of queried category and
asserted colour between the arms, not by the evidence alone.

Fabrication at 25% answered coverage (130 of 520 items): 0.185 / 0.154 / 0.146
(3B / 7B / LLaVA) for the frozen probe, against 0.254 / 0.154 / 0.054 for
p(unknown) and 0.346 / 0.169 / 0.600 for maximum probability. In 3B the gate
still beats the model's own signal within-question, by 0.069 instead of 0.115
on the main pair; in 7B it ties and in LLaVA it is worse. 7B answers only 26%
of matched conflict items, so this operating point selects 95% of its answers
and is weakly informative; LLaVA answers 90%, so its point is informative and
unfavourable. The main-design gate advantage is partly arm-diagnostic.

### 3.2 Main-design reference and gray images

| control | refit probe, no hint | refit probe, conflict | p(unknown), no hint | p(unknown), conflict | maxprob, no hint | maxprob, conflict |
|---|---|---|---|---|---|---|
| main pair (grouped by sample) | 0.995 | 0.998 | 0.993 | 0.834 | 0.615 | 0.463 |
| grouped by category | 0.995 | 0.997 | — | — | — | — |
| gray image (text only) | 0.849 | 0.981 | 0.700 | 0.325 | — | — |

The direction learned on gray images transfers across hints at 0.540 / 0.603,
versus 0.958 / 0.990 for the real-image direction on held-out scenes (0.971 /
0.996 when scored on its own fitting scenes).

## 4. Steering (3B)

In-sample direction, n=261:

| setting | present refusal | present top-1 conf. | absent refusal | absent answers colour |
|---|---|---|---|---|
| base | 0.011 | 0.836 | 0.904 | 0.096 |
| ablate w | 0.004 | 0.838 | 0.759 | 0.241 |
| -0.50 w | 0.352 | 0.741 | 0.989 | 0.011 |
| -0.25 w | 0.130 | 0.760 | 0.973 | 0.027 |
| +0.25 w | 0.000 | 0.878 | 0.330 | 0.670 |
| +0.50 w | 0.000 | 0.876 | 0.000 | 1.000 |

Held-out half (direction fitted on 130 scenes, evaluated on 131):

| setting | present refusal | absent refusal |
|---|---|---|
| base | 0.015 | 0.908 |
| ablate w | 0.008 | 0.786 |
| -0.50 w | 0.450 | 0.992 |
| -0.25 w | 0.214 | 0.985 |
| +0.25 w | 0.000 | 0.305 |
| +0.50 w | 0.000 | 0.000 |

## 4b. Steering under conflict (3B)

The same no-hint direction, injected during the conflicting pass. `true` and
`hint` are argmax rates on present items; absent `hint` is the rate of naming
the hinted colour, averaged over a and b.

| setting | present refusal | present true | present hint | absent refusal | absent hint |
|---|---|---|---|---|---|
| no hint, unsteered | 0.011 | 0.762 | — | 0.904 | — |
| conflict, base | 0.088 | 0.383 | 0.448 | 0.542 | 0.450 |
| ablate w | 0.031 | 0.429 | 0.448 | 0.328 | 0.651 |
| -0.50 w | 0.931 | 0.008 | 0.042 | 0.941 | 0.057 |
| -0.25 w | 0.759 | 0.073 | 0.134 | 0.874 | 0.125 |
| +0.25 w | 0.000 | 0.479 | 0.433 | 0.065 | 0.845 |
| +0.50 w | 0.000 | 0.479 | 0.410 | 0.000 | 0.665 |

Paired differences vs the unsteered conflict base, cluster bootstrap over
scenes (2,000 resamples):

| setting | present refusal | present true | present hint | absent refusal | absent hint |
|---|---|---|---|---|---|
| ablate w | -0.057 [-0.084, -0.031] | +0.046 [+0.019, +0.077] | 0.000 [-0.027, +0.023] | -0.215 [-0.253, -0.176] | +0.201 [+0.165, +0.236] |
| -0.50 w | +0.843 [+0.801, +0.885] | -0.375 [-0.433, -0.322] | -0.406 [-0.464, -0.345] | +0.398 [+0.354, +0.446] | -0.393 [-0.439, -0.351] |
| -0.25 w | +0.670 [+0.617, +0.728] | -0.310 [-0.368, -0.253] | -0.314 [-0.372, -0.257] | +0.331 [+0.289, +0.374] | -0.326 [-0.368, -0.285] |
| +0.25 w | -0.088 [-0.123, -0.054] | +0.096 [+0.050, +0.142] | -0.015 [-0.065, +0.034] | -0.477 [-0.523, -0.429] | +0.395 [+0.345, +0.443] |
| +0.50 w | -0.088 [-0.123, -0.054] | +0.096 [+0.034, +0.153] | -0.038 [-0.103, +0.027] | -0.542 [-0.592, -0.490] | +0.215 [+0.159, +0.270] |

Reading: the direction still gates refusal under conflict, and ablation is
still partly necessary at the readout. It does not resolve the colour: at
+0.25 w true-colour answers rise by 9.6 pp while hint adoption is unchanged
(-1.5 pp, CI spans 0). The evidence is compressed and under-weighted, not
erased. Because the vector is injected immediately before unembedding, this is
readout-level sufficiency, not evidence that the model's upstream gate uses w.
Raw run: `results/runs/steer_conflict.jsonl`.

## 4c. Steering under conflict, cross-model

The direction is refitted on each model's no-hint states with the same
construction. Base and the two non-saturated doses (LLaVA uses a 512 px longest
side, matching its main run):

| model | setting | present refusal | present true | present hint | absent refusal | absent hint |
|---|---|---|---|---|---|---|
| 3B | base | 0.088 | 0.383 | 0.448 | 0.542 | 0.450 |
| | -0.25 w | 0.759 | 0.073 | 0.134 | 0.874 | 0.125 |
| | +0.25 w | 0.000 | 0.479 | 0.433 | 0.065 | 0.845 |
| 7B | base | 0.556 | 0.249 | 0.176 | 0.918 | 0.082 |
| | -0.25 w | 0.950 | 0.011 | 0.038 | 1.000 | 0.000 |
| | +0.25 w | 0.004 | 0.421 | 0.425 | 0.057 | 0.870 |
| LLaVA | base | 0.004 | 0.360 | 0.490 | 0.138 | 0.833 |
| | -0.25 w | 0.345 | 0.149 | 0.253 | 0.734 | 0.241 |
| | +0.25 w | 0.000 | 0.379 | 0.533 | 0.000 | 0.990 |

Unsteered no-hint refusal (present/absent): 3B 0.011/0.904, 7B 0.100/0.958,
LLaVA 0.004/0.820. Paired LLaVA effects of -0.25 w: present refusal +0.341
[0.284, 0.402], absent refusal +0.596 [0.550, 0.640]; of +0.25 w: absent hint
+0.157 [0.123, 0.192], present hint +0.042 [0.004, 0.084], present true +0.019
[-0.011, 0.050]. Paired 7B effects of +0.25 w: present true +0.172
[0.111, 0.234], present hint +0.249 [0.188, 0.307], absent hint +0.787
[0.747, 0.826].

Reading: refusal is bidirectionally and monotonically controllable in all
three models, whose base policies range from over-refusal (7B) to near-total
credulity (LLaVA). Opening the gate does not choose the answer. LLaVA becomes
even more hint-adherent on absent items (0.833 to 0.990) and its present-item
true-colour rate is statistically unchanged (+0.019); 7B recovers true-colour
answers but also adopts the hint more; 3B's present-item content barely changes
(hint adoption 0.448 to 0.433, true colour 0.383 to 0.479). The direction is
an abstention gate, not an answer-selection mechanism. Raw runs: `results/runs/steer_conflict_7b.jsonl`,
`results/runs/steer_conflict_llava.jsonl`.

## 4d. Steering controls: site and direction (3B, 7B, LLaVA, conflict pass)

Dose-matched layer injection: `alpha` is 0.25 of each layer's median last-token
norm, so the perturbation is comparable across depths. Effects are paired
differences vs the unsteered conflict base (absent refusal 0.542 for 3B, 0.918
for 7B, 0.138 for LLaVA).

| model | site | -0.25 w, absent refusal | +0.25 w, absent refusal |
|---|---|---|---|
| 3B (36 layers) | L8 (22%) | -0.013 [-0.027, -0.002] | +0.008 [-0.004, +0.021] |
| | L16 (44%) | +0.019 [+0.010, +0.033] | -0.004 [-0.015, +0.008] |
| | L24 (67%) | +0.054 [+0.033, +0.075] | -0.080 [-0.103, -0.057] |
| | L32 (89%) | +0.259 [+0.220, +0.297] | -0.420 [-0.464, -0.375] |
| | lm head | +0.331 [+0.289, +0.374] | -0.477 [-0.523, -0.429] |
| 7B (28 layers) | L8 (29%) | +0.011 [+0.004, +0.021] | -0.006 [-0.013, 0.000] |
| | L14 (50%) | +0.004 [-0.006, +0.013] | -0.006 [-0.015, +0.004] |
| | L20 (71%) | +0.027 [+0.011, +0.044] | -0.015 [-0.029, -0.004] |
| | L26 (93%) | +0.056 [+0.034, +0.079] | -0.125 [-0.153, -0.096] |
| LLaVA (28 layers) | L8 (29%) | -0.010 [-0.019, -0.002] | +0.006 [-0.002, +0.015] |
| | L14 (50%) | +0.025 [+0.011, +0.042] | +0.010 [0.000, +0.023] |
| | L20 (71%) | +0.203 [+0.170, +0.238] | -0.123 [-0.153, -0.094] |
| | L26 (93%) | +0.307 [+0.270, +0.345] | -0.138 [-0.172, -0.107] |

Three random unit directions orthogonal to `w` (`r0`-`r2`), same norm,
injected at the lm head:

| model | direction | mean abs absent-refusal effect | -0.25 w, absent refusal | +0.25 w, absent refusal |
|---|---|---|---|---|
| 3B | w | 0.404 | +0.331 | -0.477 |
| | r0 | 0.061 | -0.088 | +0.034 |
| | r1 | 0.361 | +0.301 | -0.420 |
| | r2 | 0.292 | -0.349 | +0.234 |
| | ablate w | 0.215 | -0.215 [-0.253, -0.176] | — |
| | ablate r0-r2 (mean) | 0.006 | +0.002 / -0.004 / -0.012 | — |
| 7B | w | 0.471 | +0.082 | -0.860 |
| | r0 | 0.224 | +0.073 | -0.374 |
| | r1 | 0.135 | +0.050 | -0.220 |
| | r2 | 0.094 | -0.157 | +0.031 |
| | ablate w | 0.211 | -0.211 [-0.245, -0.174] | — |
| | ablate r0-r2 (mean) | 0.003 | -0.004 / +0.002 / -0.002 | — |
| LLaVA | w | 0.367 | +0.596 | -0.138 |
| | r0 | 0.152 | +0.193 | -0.111 |
| | r1 | 0.107 | +0.103 | -0.111 |
| | r2 | 0.121 | +0.132 | -0.109 |
| | ablate w | 0.102 | -0.102 [-0.130, -0.073] | — |
| | ablate r0-r2 (mean) | 0.007 | -0.002 / +0.008 / -0.010 | — |

Reading. The site profile is late in all three models: flat through L16 in 3B
and through L20 in 7B, then steep only at L32 (3B) and L26 (7B); LLaVA is flat
to L14 and rises earlier, to 0.20 at L20 and 0.31 at L26, still within its last
30% of depth. Adding a large vector is not direction-specific in any model:
`r1` reproduces most of `w`'s 3B refusal effect (+0.30/-0.42 vs +0.33/-0.48),
7B's strongest random reaches -0.37 against -0.86 for `w`, and LLaVA's `r0`
reaches +0.19/-0.11 against +0.60/-0.14 for `w`. `w` is 1.7x the mean random
effect on absent refusal in 3B, 3.1x in 7B and 2.9x in LLaVA, not unique as a
large perturbation. Removal is much larger for `w` than for these random directions: ablating `w`
moves absent refusal by 0.215 (3B), 0.211 (7B) and 0.102 (LLaVA) and hint
adoption by 0.201, 0.209 and 0.113, whereas ablating any random direction moves
every metric by at most 0.015. That ratio is not interpretable as specificity,
because the projections remove different amounts of the state: the mean
absolute projection fraction is 0.055 for `w`, against 0.019 / 0.004 / 0.008
for the random directions (5.4x). Directions aligned with class structure
remove comparable energy (class-mean 0.052, first PC 0.066), and no
equal-removal-norm ablation was run. We therefore claim no direction-specific
necessity. What survives is weaker: removing a direction that carries the
evidence ranking, at the readout, reliably opens the gate, consistent with the
readout depending on a broad low-dimensional subspace that `w` shares with
class structure. At +0.25, `+w` raises true-colour answers in 3B (+0.096) and
7B (+0.172) and is neutral in LLaVA (+0.019 [-0.011, +0.050]); `+r0` also
raises true answers (+0.061 / +0.218 / +0.031) but suppresses present-item
hint adoption in 3B (-0.153) while barely moving it in LLaVA (-0.027) and
raising it in 7B (+0.084); `+r1` opens the 3B gate without improving true
answers (-0.008). Raw runs:
`results/runs/control_steer_{3b,7b,llava}.jsonl`; analyses:
`results/control_causal_{3b,7b,llava}.json`.

## 5. Layer-wise patching

`same` pastes the sample's no-hint final-token state into the hinted pass at
layer L; `shuffled` pastes another sample's no-hint state.

| model | source | refusal | p(unknown) | recovery (p) |
|---|---|---|---|---|
| Qwen2.5-VL-3B (36 layers) | no hint | 0.904 | 0.762 | 100% |
| | conflict | 0.521 | 0.470 | 0% |
| | same L24 (67%) | 0.594 | 0.535 | 22% |
| | same L32 (89%) | 0.839 | 0.704 | 80% |
| | shuffled L24 | 0.631 | 0.544 | 25% |
| | shuffled L32 | 0.927 | 0.757 | 98% |
| Qwen2.5-VL-7B (28 layers) | no hint | 0.958 | 0.899 | 100% |
| | conflict | 0.877 | 0.783 | 0% |
| | same L18 (64%) | 0.893 | 0.782 | 0% |
| | same L24 (86%) | 0.962 | 0.863 | 69% |
| | same L26 (93%) | 0.962 | 0.888 | 90% |
| | shuffled L26 | 0.965 | 0.902 | 103% |
| LLaVA-OneVision-7B (28 layers) | no hint | 0.820 | 0.655 | 100% |
| | conflict | 0.088 | 0.134 | 0% |
| | same L18 (64%) | 0.330 | 0.324 | 36% |
| | same L24 (86%) | 0.743 | 0.549 | 80% |
| | same L26 (93%) | 0.808 | 0.633 | 96% |
| | shuffled L26 | 0.842 | 0.662 | 101% |

`shuffled` uses the previous sample's no-hint state; source and target are both
absent-object scenes. It recovers as much or more, which indicates a
category-level "no evidence" state rather than a sample-specific code, but it
also means the control cannot separate restoring the evidence state from
restoring a generic absent/no-hint prior. Patching localises the stage; it is
not an effect-size estimate.

## 5b. Same-image, opposite-question patching (3B, 7B, LLaVA)

`patch5` holds the image fixed and varies the question. Each source is a
no-hint last-token state pasted into the absent/conflicting pass at layer L:
`same_abs` is this image asked about the absent category, `same_pres` is this
image asked about the present category, `shuf_abs` / `shuf_pres` are the
previous sample's states. Refusal is over `absent|a` only (conflict base 0.521
for 3B, 0.877 for 7B, 0.088 for LLaVA); recovery is relative to (no-hint -
conflict), 0.904-0.521 for 3B, 0.958-0.877 for 7B and 0.820-0.088 for LLaVA.

| model | source | L24 refusal | L24 hint | L24 recovery | late refusal | late hint | late recovery |
|---|---|---|---|---|---|---|---|
| 3B | same_abs | 0.594 | 0.383 | +0.190 [+0.104, +0.282] | 0.839 (L32) | 0.100 | +0.830 [+0.750, +0.899] |
| | same_pres | 0.084 | 0.663 | -1.140 [-1.537, -0.868] | 0.008 (L32) | 0.142 | -1.340 [-1.778, -1.034] |
| | shuf_abs | 0.631 | 0.365 | +0.290 [+0.176, +0.408] | 0.927 (L32) | 0.008 | +1.060 [+0.937, +1.192] |
| | shuf_pres | 0.088 | 0.750 | -1.120 [-1.489, -0.839] | 0.012 (L32) | 0.119 | -1.320 [-1.744, -1.009] |
| 7B | same_abs | 0.962 | 0.015 | +1.048 [+1.000, +1.222] | 0.962 (L26) | 0.011 | +1.048 [+1.000, +1.222] |
| | same_pres | 0.123 | 0.119 | -9.381 [-18.818, -5.968] | 0.100 (L26) | 0.119 | -9.667 [-19.275, -6.156] |
| | shuf_abs | 0.973 | 0.000 | +1.190 [+0.810, +1.900] | 0.965 (L26) | 0.000 | +1.095 [+0.684, +1.765] |
| | shuf_pres | 0.146 | 0.096 | -9.048 [-19.096, -5.581] | 0.104 (L26) | 0.096 | -9.571 [-20.203, -5.937] |
| LLaVA | same_abs | 0.743 | 0.096 | +0.895 [+0.847, +0.939] | 0.808 (L26) | 0.050 | +0.984 [+0.964, +1.000] |
| | same_pres | 0.004 | 0.161 | -0.115 [-0.175, -0.065] | 0.004 (L26) | 0.146 | -0.115 [-0.175, -0.066] |
| | shuf_abs | 0.804 | 0.031 | +0.979 [+0.895, +1.072] | 0.842 (L26) | 0.008 | +1.032 [+0.955, +1.118] |
| | shuf_pres | 0.008 | 0.127 | -0.111 [-0.171, -0.063] | 0.008 (L26) | 0.119 | -0.111 [-0.171, -0.063] |

Paired contrasts in recovery units: 3B same_abs - same_pres = +0.510
[+0.448, +0.567] at L24 and +0.831 [+0.785, +0.874] at L32; same_abs -
shuf_abs = -0.038 [-0.081, +0.004] at L24 and -0.088 [-0.142, -0.038] at L32.
7B same_abs - same_pres = +0.839 [+0.789, +0.885] at L24 and +0.862
[+0.816, +0.904] at L26; same_abs - shuf_abs = -0.012 [-0.042, +0.019] at L24
and -0.004 [-0.038, +0.027] at L26. LLaVA same_abs - same_pres = +0.739
[+0.682, +0.793] at L24 and +0.805 [+0.755, +0.851] at L26; same_abs -
shuf_abs = -0.062 [-0.131, +0.012] at L24 and -0.035 [-0.092, +0.027] at L26.

Reading: at the late site the pasted state transfers the policy of its own
question in all three models. An absent-question state restores refusal to
near the no-hint level; a present-question state drives refusal toward zero,
whether or not it comes from the same image. The same_abs - same_pres gap
separates the two question-conditioned policies (+0.74 to +0.86 in raw refusal
at the late site), and its sign and size are stable across models with very
different base policies (7B over-refuses at 0.88, LLaVA answers at 0.09). The
negative same_abs - shuf_abs contrast shows the state is not sample-specific:
another absent scene restores refusal at least as well, and in LLaVA the
confidence interval includes zero at both layers. The late state therefore
carries a question-conditioned answer/refuse policy and a generic absent-scene
status, not a code for this image's evidence. Caveat: 7B's abstention is
already near ceiling under conflict, so its span is only 0.081 and its recovery
ratios are unstable by construction; the raw contrast is the interpretable
quantity there.

## 6. Selective gate: fabrication at matched answered coverage under conflict

Deployment numbers use scene-cross-fitted probe scores: the probe is fitted on
the no-hint pairs of four scene folds and applied to the conflict pairs of the
held-out fold, so no evaluation scene contributes a label to the fit. The
plausible hint pair is `sufficient|wrong` vs `absent|a`. Coverage is the
fraction of all items the system answers after gating; the threshold is chosen
per scorer to reach a target answered coverage, so gates are compared at the
same achieved coverage rather than at the same number of selected items.

An earlier version reported risk among answered at 50% *selection* coverage.
That protocol is degenerate when a score ranks confident refusals highly:
7B max-probability selects half the items but answers only 45 of them, and 7B
answers only 28.4% of conflict items in total, so a 50% operating point does
not exist for 7B. The matched-coverage numbers below supersede the earlier
table.

| model | target | frozen (OOF) | refit | p(unknown) | maxprob (11 colours) |
|---|---|---|---|---|---|
| 3B | 25% | 0.077 | 0.046 | 0.192 | 0.315 |
| 3B | 50% | 0.115 | 0.100 | 0.249 | 0.368 |
| 7B | 25% | 0.146 | 0.131 | 0.169 | 0.215 |
| 7B | 50% | infeasible (max 0.284) | — | — | — |
| LLaVA | 25% | 0.062 | 0.031 | 0.138 | 0.762 |
| LLaVA | 50% | 0.149 | 0.107 | 0.295 | 0.655 |

At the shared 25% point the cross-fitted gate reduces fabrication by 2.5x in
3B and 2.2x in LLaVA relative to `p(unknown)`; in 7B the nominal ratio is 1.2x
but not distinguishable from zero. Scene bootstrap of the paired difference at
25% coverage (OOF scores held fixed, 500 draws; `pilot/gate_ci.py`): 3B probe
0.077 [0.031,0.115] vs p(unknown) 0.192 [0.123,0.254], difference [-0.169,-0.069];
7B 0.146 [0.092,0.215] vs 0.169 [0.108,0.231], difference [-0.062,+0.008];
LLaVA 0.062 [0.023,0.115] vs 0.138 [0.092,0.200], difference [-0.138,-0.008].
At 50% the gain is 2.2x in 3B and 2.0x in LLaVA; 7B never reaches that coverage
because it over-refuses under conflict. The insample/refit column is not deployable and
is shown only as the decodability ceiling. A logistic gate over the 12 output
probabilities (3B) fabricates 0.142 at 50% selection coverage, so the gain is
not merely a learned readout of the output distribution.

Prevalence shift, threshold calibrated at 25% answered coverage on the balanced
set, fabrication among answered (probe vs `p(unknown)`):

| unanswerable prevalence | 5% | 10% | 25% | 50% |
|---|---|---|---|---|
| 3B probe (p(unknown)) | 0.008 (0.023) | 0.017 (0.048) | 0.050 (0.131) | 0.137 (0.312) |
| 7B probe (p(unknown)) | 0.032 (0.037) | 0.064 (0.076) | 0.171 (0.198) | 0.383 (0.425) |
| LLaVA probe (p(unknown)) | 0.004 (0.009) | 0.008 (0.019) | 0.023 (0.055) | 0.067 (0.149) |

Conformal risk control is underpowered at this sample size and we do not claim
certification. With the answered subset in both numerator and denominator
(`risk among answered`), 130 calibration scenes and a Clopper-Pearson 90% UCB
over a 50-point threshold scan with family-wise correction, no half-split is
feasible at alpha=5% or alpha=10% in any model (0/20 splits; minimum UCB 0.174
/ 0.210 / 0.191 in 3B / 7B / LLaVA). Without the family-wise correction the
10% bound is pointwise feasible in 7/20 (3B), 11/20 (7B) and 7/20 (LLaVA)
splits at answered coverage 0.30 / 0.19 / 0.22, and the held-out risk is 0.088
/ 0.060 / 0.107, so LLaVA still misses target. The earlier claim that a 10%
certificate holds for two of three models came from an inconsistent denominator
(refused absent items counted in the denominator but not the numerator) and
from selecting a favourable split; both are corrected here.

A pollution detector `probe - p(unknown)` is not useful: present-vs-absent
AUROC is 0.97 / 0.95 / 0.96, but it selects nearly the same items as the probe
alone.

Always-on and fixed-threshold checks (3B). Pooled over all three hint
conditions, ranking by the no-hint probe gives 0.064 fabrication at 50%
*selection* coverage versus 0.193 for p(known)=1-p(unknown); at 70% selection
coverage, 0.400 versus 0.480.
A fixed threshold at the probe's natural boundary (z>0) answers 43.4% of
pooled items with 6.6% fabrication and 7.5% false refusal; under conflict it
answers only 32.7% with 7.9% fabrication and 17.6% false refusal. The
contraction of the axis makes a fixed threshold conservative under conflict;
coverage-based gating is rank-invariant and does not have this failure mode.

## 7. Task-form boundaries (3B)

Probe values in this section are refit inside each condition (upper bound);
frozen transfer is reported in Sections 2 and 9.

| task | condition | refusal | adopts hint | probe | p(unknown) | maxprob |
|---|---|---|---|---|---|---|
| colour | present, no hint | 0.011 | — | 0.995 | 0.993 | 0.615 |
| colour | present, conflict | 0.088 | 0.448 | 0.997 | 0.834 | 0.463 |
| colour | absent, no hint | 0.904 | — | — | — | — |
| colour | absent, hint | 0.521 | 0.467 | — | — | — |
| side | present, no hint | 0.000 | — | 0.993 | 0.989 | 0.791 |
| side | present, conflict | 0.000 | 0.268 | 0.991 | 0.966 | 0.744 |
| side | absent, no hint | 0.621 | — | — | — | — |
| side | absent, hint | 0.111 | 0.743 | — | — | — |
| count | present, no hint | 0.682 | — | 0.996 | 0.986 | 0.014 |
| count | present, conflict | 0.870 | 0.115 | 0.995 | 0.946 | 0.055 |
| count | absent, no hint | 1.000 | 0.000 | — | — | — |
| count | absent, hint | 0.996 | 0.004 | — | — | — |

## 8. External validation (frozen probe)

VizWiz val, 462 items (212 answerable, 250 unanswerable):

| arm | condition | refusal | accuracy | p(unknown) | maxprob |
|---|---|---|---|---|---|
| answerable | no hint | 0.099 | 0.580 | 0.111 | 0.544 |
| answerable | correct hint | 0.033 | 0.816 | 0.052 | 0.650 |
| answerable | conflicting hint | 0.193 | 0.451 | 0.214 | 0.459 |
| unanswerable | no hint | 0.704 | 0.087 | 0.520 | 0.598 |
| unanswerable | hint | 0.636 | 0.041 | 0.492 | 0.554 |

Frozen probe AUROC for evidence: 0.827 [0.789, 0.862] no hint; 0.720
[0.671, 0.766] with a hint. p(unknown): 0.878 / 0.793. Risk at 50% coverage:
probe 0.457, maxprob 0.655, p(unknown) 0.449.

VQA-v2 val, 300 colour questions:

| condition | accuracy | p(unknown) | maxprob | probe mean | probe > 0 | pred. unknown | pred. distractor |
|---|---|---|---|---|---|---|---|
| no hint | 0.850 | 0.024 | 0.865 | +6.932 | 0.940 | 0.017 | 0.010 |
| correct hint | 0.983 | 0.021 | 0.971 | +6.698 | 0.983 | 0.017 | 0.000 |
| conflicting hint | 0.613 | 0.115 | 0.722 | +2.100 | 0.780 | 0.057 | 0.280 |

Probe stability between no-hint and conflict: r = 0.537; mean shift -4.831.
AUROC for answer correctness: no-hint score 0.710, conflict score 0.489;
maxprob 0.831 / 0.649. Risk at 50% coverage: probe 0.087, maxprob 0.007.

## 9. Cross-model summary

| model | probe, no hint | frozen probe, conflict | refit probe, conflict | p(unknown), no hint | p(unknown), conflict | maxprob, conflict | 12-d logit probe, conflict | gap ratio b [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 0.995 | 0.958 | 0.995 | 0.993 | 0.834 | 0.463 | 0.923 | 0.463 [0.438, 0.488] |
| Qwen2.5-VL-7B | 0.995 | 0.946 | 0.993 | 0.984 | 0.797 | 0.306 | 0.896 | 0.424 [0.393, 0.457] |
| LLaVA-OneVision-7B | 0.994 | 0.935 | 0.998 | 0.991 | 0.781 | 0.289 | 0.909 | 0.344 [0.324, 0.365] |
| Gemma-3-4B-it | 0.986 | 0.964 | 0.989 | 0.961 | 0.891 | 0.701 | — | 0.543 [0.502, 0.580] |
| Phi-3.5-vision | 0.992 | 0.685 | 0.998 | 0.939 | 0.385 | — | — | 0.149 [0.096, 0.212] |

The frozen column is the deployable headline (scene-held-out, hint b, C=0.1);
the refit column is an upper bound from retraining inside the conflict
condition. Hint a gives 0.957 / 0.939 / 0.923 frozen. Two backbones sit outside
the Qwen family: Gemma-3-4B-it (Gemma-3 language model + SigLIP) has frozen
values 0.964 (hint a) and 0.970 (hint b) with refits 0.989 / 0.990, and its
maxprob *rises* under conflict; Phi-3.5-vision (Phi-3 + CLIP) has frozen 0.685
(hint a) / 0.675 (hint b) with a refit of 0.998, and its p(unknown) falls to
0.385 / 0.458. The output-collapse columns therefore fail only for Gemma.

## 10. Revision controls (3B / 7B, LLaVA where run)

All controls use the same 261-item manifest and the same prompt battery; only
the intervention changes.

### 10.1 Energy-matched removal

Projecting `w` out of the final state removes 5.4x more absolute energy than a
random direction (mean projection fraction 0.055 vs 0.019 / 0.004 / 0.008), so
the raw ablation conflates direction with removed energy. Rescaling every
removal to a common target energy (`k` = target/self) separates them:

| model | base | w raw | w -> r | r -> w | cm -> w | pc1 -> w |
|---|---|---|---|---|---|---|
| 3B | 0.542 | 0.328 [-.253,-.180] | 0.469 [-.096,-.052] | 0.552 [-.004,+.025] | 0.625 [+.052,+.111] | 0.625 [+.052,+.111] |
| 7B | 0.918 | 0.707 [-.247,-.178] | 0.900 [-.031,-.008] | 0.874 [-.063,-.027] | 0.921 [-.011,+.019] | 0.921 [-.011,+.019] |
| LLaVA | 0.138 | 0.036 [-.130,-.075] | 0.126 [-.021,-.004] | 0.109 [-.046,-.013] | 0.075 [-.088,-.042] | 0.100 [-.057,-.021] |

Base is absent refusal under the conflicting pass, averaged over hints a and b;
brackets are 95% scene-bootstrap intervals on the change from base. At matched
energy no direction is necessary in a direction-specific sense: w's residual
effect is smaller than a matched random direction in 7B, and class-mean / PC1
directions (cosine .944 / .997 with w) move 3B refusal more than w does. In
LLaVA, where base refusal is already near the floor, w's matched effect
(-0.011) is smaller than both the matched random (-0.029) and matched
class-mean (-0.063) effects; its projection energy is 5.3x the mean random
energy. The claim is readout-level sufficiency plus an efficient direction, not
a circuit.

### 10.2 Patching is readable (no-op control)

`patch6.py` pastes each pass's own captured state back at a layer through the
same hook. If the hook is wired correctly the distribution must be identical:
argmax agreement is 1.000 and max |dp| is 0.0 at L24/L32 (3B), L18/L26 (7B) and
L18/L26 (LLaVA). Pasting zeros through the same hook changes the argmax on
10.7% / 17.2% (3B), 38.7% / 6.1% (7B) and 72.4% / 99.6% (LLaVA) of items, so the
hook is active and the earlier patch effects are not vacuous.

### 10.3 Prompt format

| model | format | refuse, none | refuse, a | refuse, b | adopt, a | adopt, b | acc, wrong |
|---|---|---|---|---|---|---|---|
| 3B | std | 0.904 | 0.521 | 0.563 | 0.467 | 0.433 | 0.383 |
| 3B | no unknown | 0.000 | 0.000 | 0.000 | 0.935 | 0.908 | - |
| 3B | sentence hint | 0.931 | 0.958 | 0.981 | 0.038 | 0.011 | 0.594 |
| 7B | std | 0.958 | 0.877 | 0.958 | 0.123 | 0.042 | 0.249 |
| 7B | no unknown | 0.000 | 0.000 | 0.000 | 0.977 | 0.973 | - |
| 7B | sentence hint | 0.958 | 0.989 | 0.992 | 0.004 | 0.000 | 0.498 |
| LLaVA | std | 0.820 | 0.088 | 0.188 | 0.904 | 0.762 | 0.360 |
| LLaVA | no unknown | 0.000 | 0.000 | 0.000 | 0.992 | 0.946 | - |
| LLaVA | sentence hint | 0.793 | 0.559 | 0.736 | 0.333 | 0.088 | 0.548 |
| Gemma | std | 0.843 | 0.402 | 0.475 | 0.579 | 0.502 | 0.241 |
| Gemma | no unknown | 0.000 | 0.000 | 0.000 | 0.789 | 0.640 | - |
| Gemma | sentence hint | 0.812 | 0.851 | 0.870 | 0.038 | 0.027 | 0.559 |
| Phi-3.5 | std | 0.897 | 0.322 | 0.425 | 0.663 | 0.567 | 0.080 |
| Phi-3.5 | no unknown | 0.004 | 0.000 | 0.000 | 0.985 | 0.950 | - |
| Phi-3.5 | sentence hint | 0.931 | 0.877 | 0.931 | 0.088 | 0.046 | 0.276 |

Removing the abstention token removes abstention by construction and raises
hint adoption; a sentence-form hint is not treated as an assertion and the
conflict effect disappears. The same pattern holds in all five models: LLaVA
moves from a lower baseline (conflict refusal 0.088 -> 0.559, adoption
0.904 -> 0.333, wrong-hint accuracy 0.360 -> 0.548), Gemma from 0.402 -> 0.851
(adoption 0.579 -> 0.038, accuracy 0.241 -> 0.559), and Phi-3.5 from
0.322 -> 0.877 (adoption 0.663 -> 0.088, accuracy 0.080 -> 0.276). Removing
unknown raises adoption in Gemma 0.579 -> 0.789 and in Phi 0.663 -> 0.985.
Both are stated as scope conditions, not as robustness checks that pass.
Gemma and Phi raw runs: `results/runs/prompt_variants_{gemma,phi}.jsonl`;
analyses `results/prompt_{gemma,phi}.json` (the Phi run was added later and
matches this table exactly).

### 10.4 Strict absence

| model | mean p(yes) | p(yes) > 0.5 | n strict | refuse, hint (all -> strict) | probe AUROC (strict) |
|---|---|---|---|---|---|
| 3B | 0.046 | 7/261 | 254 | 0.521 -> 0.531 | 0.999 |
| 7B | 0.014 | 2/261 | 259 | 0.877 -> 0.884 | 0.997 |
| LLaVA | 0.403 | 73/261 | 183 | 0.088 -> 0.104 | 1.000 |

Restricting to items the presence probe judges absent leaves refusal, adoption
and probe AUROC unchanged, so annotation absence is not doing the work. For
LLaVA the presence check itself is much weaker (mean p(yes) 0.403, 28% judged
present above 0.5): its yes/no readout is not a reliable absence detector, so
for LLaVA "absent" rests on the COCO annotation, not on the probe. The conflict
result on the strict subset is nevertheless unchanged (0.088 -> 0.104, adoption
0.891, probe 1.000).

### 10.5 Scene fit of the absent category

Absent categories are sampled without requiring scene plausibility. Scoring each
by COCO train co-occurrence with the objects annotated in the image,
s = mean_c P(absent|c)/P(absent), and splitting at the median:

| model | half | n | refuse, none | refuse, a | adopt, a | AUROC / probe |
|---|---|---|---|---|---|---|
| 3B | lo | 131 | 0.962 | 0.603 | 0.389 | 0.998 / 0.999 |
| 3B | hi | 130 | 0.846 | 0.438 | 0.546 | 0.988 / 0.992 |
| 7B | lo | 131 | 0.992 | 0.901 | 0.099 | 0.991 / 1.000 |
| 7B | hi | 130 | 0.923 | 0.854 | 0.146 | 0.976 / 0.989 |
| LLaVA | lo | 131 | 0.885 | 0.076 | 0.916 | 0.995 / 0.997 |
| LLaVA | hi | 130 | 0.754 | 0.100 | 0.892 | 0.986 / 0.993 |

Better-fitting absent categories are refused less (hi - lo: -0.116
[-0.185,-0.047] 3B, -0.069 [-0.115,-0.023] 7B, -0.132 [-0.224,-0.039] LLaVA).
Within both strata p(unknown) still separates present from absent (AUROC at
least 0.976) and the probe at least 0.989: scene fit moves the level of the
gate, not the ranking of evidence.

Category frequency is the obvious rival explanation. Splitting on the absent
category's COCO train image count instead, frequent categories are refused less
in 7B (-0.073 [-0.128,-0.025]) and LLaVA (-0.100 [-0.195,-0.008]) but not
detectably in 3B (-0.062 [-0.133,+0.014]). The two scores are only weakly
related (Spearman 0.225). A logistic fit of no-hint refusal on both, standardised
and bootstrapped over scenes (1000 draws), leaves scene fit negative in every
model, with frequency contributing only in 7B:

| model | scene fit (per SD) | frequency (per SD) |
|---|---|---|
| 3B | -0.808 [-1.260,-0.370] | -0.286 [-0.763,+0.109] |
| 7B | -0.683 [-1.240,-0.291] | -0.454 [-0.976,-0.137] |
| LLaVA | -0.438 [-0.775,-0.156] | -0.010 [-0.340,+0.367] |

No-hint refusal in the 2x2 of scene fit (rows) and frequency (columns), cell
sizes 80 / 51 / 56 / 74: 3B 0.963 / 0.961 / 0.893 / 0.811, 7B 1.000 / 0.980 /
0.982 / 0.878, LLaVA 0.887 / 0.882 / 0.839 / 0.689. The scene-fit drop appears
within both frequency strata, so frequency does not carry the effect.

### 10.6 Input resolution

3B rerun at 262,144 and 589,824 pixels: absent refusal 0.908 / 0.904 with no
hint, 0.556 / 0.521 under hint a, adoption 0.441 / 0.467, wrong-hint accuracy
0.433 / 0.383, frozen probe 0.997 / 1.000. Resolution moves the level by at
most 0.05 and does not create the conflict effect.

### 10.7 Per-layer transfer with a single estimator

One logistic-regression estimator is fitted on the final layer per CV fold and
applied unchanged to every earlier layer (per-layer standardisation only).
Transfer AUROC is at chance through position 18 (<=0.53), 0.58 / 0.64 at 19 /
21, 0.86 at 22, 0.67 at 23, 0.97 at 24 and >=0.99 from 25 on. Refitting per
layer is decodable from layer 16 (>=0.99). Evidence is thus present early in a
layer-specific basis, while the final-layer axis is assembled late - which is
why patching only recovers in the last few layers.

### 10.8 A model without an abstention behaviour (SmolVLM)

SmolVLM-Instruct (2B; Idefics3 vision encoder + SmolLM2) never emits `unknown`:
refusal is 0 in all six cells and it adopts the stated colour on 0.985 (hint a)
/ 0.989 (hint b) of absent items. Accuracy is 0.655 with no hint, 0.973 with the
correct hint and 0.276 under the conflicting hint. There is no behavioural gate
to study, so it is not a fourth replication. Its final state still separates
present from absent: probe AUROC 0.989 (no hint), 0.993 (conflict), 0.993
(correct); p(known) 0.921 / 0.598 / 0.760; max probability 0.877 / 0.257 / 0.555;
frozen transfer from the no-hint pair 0.924 (conflict) and 0.967 (correct). A
decodable evidence axis can exist without abstention. Caveat: on the conflict
pair the probe could partly read hint-image mismatch, so this is a boundary
observation about the axis, not about the gate.

### 10.9 A backbone with a gate outside the Qwen family (Gemma-3-4B)

Gemma-3-4B-it (Gemma-3 language model + SigLIP encoder, run at a 512-pixel
resize in bf16, 261 items x 6 conditions, about 6 minutes on the A10) is the
first backbone here whose language model is not from the Qwen family. It does
abstain, so unlike SmolVLM it can test the gate.

| condition | refusal | mean conf. | adopts hint | accuracy |
|---|---|---|---|---|
| present, correct hint | 0.004 | 0.992 | — | 0.943 |
| present, conflicting hint | 0.008 | 0.974 | 0.674 | 0.241 |
| present, no hint | 0.050 | 0.966 | — | 0.567 |
| absent, hint a | 0.402 | 0.928 | 0.579 | — |
| absent, hint b | 0.475 | 0.955 | 0.502 | — |
| absent, no hint | 0.843 | 0.978 | — | — |

The behaviour has the same shape as 3B: the hint moves both arms toward the
assertion (absent refusal 0.843 -> 0.402/0.475, adoption 0.579/0.502; present
accuracy 0.567 -> 0.241 at adoption 0.674). Its no-hint colour accuracy is
lower (0.567 against 0.762 in 3B), so the present arm is noisier.

Internal signals (probe CV by scene, C=0.1; frozen transfer = fit on training
scenes' no-hint states, score held-out scenes):

| pair | probe | p(unknown) | maxprob | frozen held-out [95% CI] | refit |
|---|---|---|---|---|---|
| none | 0.986 | 0.961 | 0.627 | — | — |
| hint a | 0.989 | 0.891 | 0.701 | 0.964 [0.950, 0.976] | 0.989 |
| hint b | — | — | — | 0.970 [0.957, 0.981] | 0.990 |
| correct | 0.991 | 0.928 | 0.868 | 0.981 [0.971, 0.989] | 0.991 |

Geometry (same estimator and noise-floor control as section 2c):

| condition | ratio | angle | along w | Mahalanobis | coherence |
|---|---|---|---|---|---|
| none (noise floor) | 1.001 [0.942, 1.060] | 4.5° [3.6, 5.6] | 1.00 | 0.39 | 0.594 |
| conflict a | 0.529 [0.490, 0.567] | 28.6° [26.5, 30.8] | 0.466 | 0.424 | 0.544 |
| conflict b | 0.543 [0.502, 0.580] | 24.3° [22.6, 26.3] | 0.496 | 0.419 | 0.563 |
| correct | 0.538 [0.501, 0.576] | 30.1° [28.1, 32.2] | 0.467 | 0.432 | 0.567 |

cos(none, conflict) 0.881, cos(none, conflict_b) 0.914, cos(conflict a,
conflict b) 0.971 — the two hinted colours rotate to almost the same direction,
more tightly than in any Qwen-family model (0.908-0.915).

Reading. Dilution (0.53), rotation (24-29°, five to six times the 4.5° noise
floor), transfer (0.964/0.970) and the colour-independent direction all
replicate outside the Qwen family. Two things do not: per-scene coherence barely
falls (0.594 -> 0.544/0.563, against 0.60 -> 0.40 in 3B), and the native output
signals do not collapse — p(unknown) AUROC falls only 0.961 -> 0.891 and
maxprob *rises* 0.627 -> 0.701 while the answers are overridden as strongly as
anywhere in the battery. The representational claim (dilution + rotation, with
the evidence still decodable and transferable) generalises across language-model
families; the rank-scale/output dissociation that motivates the frozen gate
appears in the Qwen2/2.5 and Phi-3 readouts but not in Gemma, so it is
readout-specific rather than tied to one family. In Gemma the gate's margin
over p(unknown) is smaller (0.964 vs 0.891) but still present. A refit gate in
the conflict condition reaches fabrication 0.038 at 50% selection coverage
against 0.172 for p(unknown), so the gate is still useful there. (The Gemma
screen used the selection-coverage convention; the cross-fitted matched-coverage
protocol was run for the three white-box models only.)

Binary SDT on the fixed no-hint axis (log-linear-corrected, 2000 scene
bootstrap draws):

| condition | d' | c | Δd' [95% CI] | Δc [95% CI] |
|---|---|---|---|---|
| none | 2.63 | +0.31 | — | — |
| conflict | 2.19 | +1.25 | -0.44 [-0.88, +0.17] | +1.25 [+0.72, +1.26] |
| correct | 3.53 | +0.76 | +0.90 [+0.53, +1.39] | +0.45 [+0.27, +0.70] |

Unlike 3B (d' 3.51 -> 1.45), Gemma shows a criterion shift without a detectable
sensitivity loss on this coarse statistic, which is consistent with its milder
rotation (24-29 deg) leaving the fixed axis comparatively intact. The
decision-level decomposition is therefore model-specific too; only the geometry
survives in every backbone tested here (section 10.10 adds a fifth).

Raw run and analyses are archived: `results/runs/v5_261_gemma.jsonl` + `.npz`,
the four analysis logs, `results/geometry_gemma.json` and
`results/sdt_binary_gemma.json`. The tables above quote that output; the run
takes about 6 minutes to regenerate with `scripts/run_nonqwen.sh`. The
screen run used to pick the model (`results/runs/screen_gemma40.*`) is archived
as well.

### 10.10 A second backbone outside the Qwen family (Phi-3.5-vision)

Phi-3.5-vision-instruct (Phi-3-mini language model + CLIP ViT-L/14-336 encoder,
3072-d hidden state, native `<|image_1|>` prompt, eager attention because the
config pins flash-attention-2, 512-pixel resize) runs the same 261 items x 6
conditions in about 8 minutes on the A10. It is the third language-model family
(Phi) and the third vision encoder (CLIP) in the battery.

| condition | refusal | gold | adopts hint |
|---|---|---|---|
| present, no hint | 0.119 | 0.625 | — |
| present, correct hint | 0.092 | 0.858 | 0.858 |
| present, conflicting hint | 0.548 | 0.080 | 0.360 |
| absent, no hint | 0.897 | — | — |
| absent, hint a | 0.322 | — | 0.663 |
| absent, hint b | 0.425 | — | 0.567 |

The hint moves both arms: absent refusal 0.897 -> 0.322/0.425 with adoption
0.663/0.567, and present accuracy 0.625 -> 0.080 under conflict. The gate is
therefore expressed in this model and can be attacked.

Internal signals (scene-grouped CV, C=0.1; frozen = fit on no-hint training
scenes, score held-out scenes):

| pair | probe | p(unknown) | maxprob | frozen held-out [95% CI] | refit |
|---|---|---|---|---|---|
| none | 0.992 [0.987, 0.997] | 0.939 | 0.442 | — | — |
| hint a | — | 0.385 | 0.492 | 0.685 [0.640, 0.731] | 0.998 |
| hint b | — | 0.458 | 0.492 | 0.675 [0.633, 0.718] | 0.998 |
| correct | — | — | — | 0.954 [0.936, 0.971] | — |

The no-hint probe and the conflict refit are 0.992 and 0.998; the frozen
transfer loses a quarter of its ranking (0.685/0.675), the lowest in the
battery, while a probe refit on the same states recovers 0.998. The model's own
signal collapses through chance (0.939 -> 0.385/0.458), as in the Qwen models
and unlike Gemma.

Geometry (same estimator as section 2c):

| condition | ratio | angle | along w | coherence |
|---|---|---|---|---|
| none | 1.00 | 0° (reference) | 1.00 | 0.388 |
| conflict a | 0.052 [0.006, 0.113] | 85.9° [80.9, 89.5] | 0.004 | 0.027 [0.008, 0.055] |
| conflict b | 0.149 [0.096, 0.212] | 74.4° [67.4, 80.2] | 0.040 | 0.070 [0.045, 0.100] |
| correct | 0.525 [0.470, 0.575] | 52.8° [49.4, 56.8] | 0.317 | 0.234 [0.215, 0.256] |

cos(conflict a, conflict b) 0.720 [0.646, 0.790]. Rotation reaches 86°, nearly
orthogonal to the no-hint axis, and per-scene coherence falls from 0.388 to
0.027/0.070. The two hinted colours are less aligned than in the other models
(0.72 against 0.89-0.97), so direction here is more colour-dependent.

Reading. Phi-3.5 is the strongest version of the dissociation: the evidence
vector is nearly annihilated along the original axis (projection 0.004/0.040),
so a frozen gate collapses to 0.685/0.675, yet the same states remain linearly
decodable at 0.998 when the probe is refit, and the output signal falls below
chance. The geometry and the output collapse both survive a third language-model
family and a third encoder, while Gemma shows the geometry without the collapse;
the collapse is therefore a property of the readout, not of a family.

Construct and predictive validity for Phi (`results/validity_phi.json`, same
scripts as section 10.8): the no-hint probe separates present from absent at
0.992 [0.986, 0.996], including present items the model answers wrongly (0.987
[0.972, 0.997]), and is much weaker within the present arm (0.680 [0.613,
0.747]), so the separation is not answerability. Predictive validity fails under
conflict: the frozen score is below chance for absent refusal (AUROC 0.392
[0.324, 0.463]), present gold (0.376 [0.241, 0.520]) and present adoption
(0.440 [0.374, 0.514]), though it predicts no-hint absent refusal (0.868
[0.797, 0.932]). This matches the weak frozen transfer rather than contradicting
it: the no-hint direction has little projection on the conflict states, so its
per-item ordering there is not usable.

Binary SDT on the fixed no-hint axis:

| condition | d' | c | Δd' [95% CI] |
|---|---|---|---|
| none | 2.43 [2.18, 2.71] | -0.04 [-0.17, +0.10] | — |
| conflict | -0.44 [-0.63, -0.26] | +0.10 [+0.01, +0.21] | -2.87 [-3.19, -2.58] |
| correct | 2.57 [2.30, 2.87] | +0.03 [-0.11, +0.18] | +0.15 [-0.12, +0.38] |

The conflict condition is worse than chance on the fixed axis (d' < 0): the
model's own signal ranks absent items as more present than present ones.

Selective gate under conflict (scene-cross-fitted probe, matched answered
coverage):

| coverage | probe | p(unknown) | maxprob | paired diff vs p(unknown) |
|---|---|---|---|---|
| 25% | 0.285 [0.208, 0.377] | 0.562 [0.477, 0.631] | 0.662 [0.577, 0.723] | -0.277 [-0.354, -0.192] |
| 50% | 0.556 [0.498, 0.609] | 0.586 [0.544, 0.632] | 0.636 [0.594, 0.678] | -0.034 [-0.073, -0.004] |

The probe still wins at 25% (fabrication 0.285 against 0.562) but the margin
narrows at 50% and the paired interval only just excludes zero. The gate
direction survives, but the frozen score is much weaker after an 86° rotation.

Dilution law: the three cells give (gap, d'/d'0) = (0.525, 0.602), (0.052,
0.049) and (0.149, 0.136); the through-origin slope is 1.13 with RMSE 0.096
against the fitted 1.425. Residuals are -0.024, -0.076 and -0.145. Phi-3.5 was
not in the 12-cell fit and its gain lies below the 1.25-1.60 range of the four
fitted models, which is what "model-specific gain" means: the linear form
survives, the constant does not extrapolate.

Raw run and analyses are archived as `results/phi_results.json` and
`results/phi_extra.json`; `scripts/run_phi.sh` regenerates them.

### 10.11 A second stimulus set (PASCAL VOC 2012, 3B)

The same design is rebuilt on PASCAL VOC 2012 val (180 images, long side
<= 1024, automatic labels only, 1080 states of 2048 dimensions; about 0.7 s per
sample) and analysed with the identical protocol and the same 3B checkpoint.

| condition | refusal | gold | adopts hint |
|---|---|---|---|
| present, no hint | 0.011 | 0.422 | — |
| present, correct hint | 0.083 | 0.839 | 0.839 |
| present, conflicting hint | 0.400 | 0.217 | 0.217 |
| absent, no hint | 0.894 | — | — |
| absent, hint a | 0.644 | — | 0.339 |
| absent, hint b | 0.683 | — | 0.311 |

The gate reproduces (absent refusal 0.894 -> 0.644/0.683) and the hint is
adopted (0.339/0.311), but no-hint colour accuracy is lower than on COCO
(0.422 against 0.762), so the present arm is noisier.

Signals:

| pair | probe | frozen held-out [95% CI] | refit |
|---|---|---|---|
| none | 0.994 [0.987, 0.999] | — | — |
| hint a | — | 0.925 [0.895, 0.955] | 0.998 |
| hint b | — | 0.950 [0.921, 0.975] | 0.998 |
| correct | — | 0.983 [0.971, 0.992] | — |

Geometry: gap ratio 0.412 [0.380, 0.446] at 34.7° [32.6, 38.4] for hint a and
0.519 [0.489, 0.550] at 33.4° [30.7, 36.7] for hint b; coherence falls
0.613 -> 0.313/0.394; cos(a, b) 0.890 [0.837, 0.922]. The rotation is smaller
than in COCO 3B (39-41°) and the frozen transfer correspondingly higher.

SDT: d' 3.44 [3.03, 4.20] with no hint, 0.67 [0.44, 0.90] under conflict
(Δ -2.77 [-3.52, -2.35]); criterion +0.48 -> -0.08.

Gate at 25% coverage: probe 0.144 [0.078, 0.211] against 0.367 [0.272, 0.444]
for p(unknown) and 0.422 [0.356, 0.517] for maxprob; paired difference
[-0.300, -0.133]. At 50% coverage only 47.8% of conflict items are answered, so
all three scores select the same set and the comparison is uninformative.

Law: cells (0.614, 0.699), (0.412, 0.499), (0.519, 0.656); through-origin slope
1.20, RMSE 0.123 against 1.425. The dilution-to-sensitivity regularity therefore
holds on a different stimulus distribution with a slightly lower gain.

Raw run and analyses: `results/voc_results.json`, `results/voc_extra.json`;
`scripts/run_voc.sh` regenerates them.

## 11. Novelty: what the data support, and what they reject

We tested two candidate quantitative laws that would link evidence geometry to
behaviour. Both fail. A weaker proportional relation does survive, and its
scope is the useful result.

1. **A single shared slope across conditions.** A logistic model of
   P(correct) on the best pre-hint evidence score, with one slope for all three
   hint conditions, is rejected in 3B (likelihood ratio 37.8, df 5,
   p = 4.2e-7). Projection-based slopes differ by cell and by model (3B none
   -0.216 vs conflict -0.037; Gemma none -0.201 vs conflict -0.742), so there
   is no condition-invariant one-slope law.

2. **Isotropic contraction.** If the hint scaled the class-conditional
   distributions isotropically, the within-class SD along the readout would
   shrink by the same factor as the mean gap. It does not
   (`results/law_isotropic.json`, 4 models x 3 hinted conditions): the gap
   ratio ranges 0.20-0.69, the SD ratio only 0.62-0.83, and a log-log fit of
   SD ratio on gap ratio has slope 0.006 (R^2 = 0.001). The readout's d' ratio
   therefore falls sub-proportionally to dilution (mean 0.60, range 0.28-1.00),
   which is why rank-based measures survive better than absolute confidence.

3. **A quantitative regularity that does survive.** For the fixed no-hint
   readout, the rank-sensitivity ratio is proportional to the measured mean-gap
   ratio: d'_c/d'_0 = 1.42 (gap_c/gap_0) + 0.003, R^2 = 0.918 over the 12
   model-condition cells (`results/law_isotropic.json`). Through the origin the
   slope is 1.425, with RMSE 0.060 and maximum residual 0.098. The gain is the
   reciprocal of the within-class spread contraction, which is nearly
   condition-independent (SD ratio 0.707 +- 0.065, range 0.62-0.83). It is
   model-specific but stable within a model across hints: 1.25 in 3B
   (1.21-1.28), 1.46 in 7B (1.43-1.49), 1.40 in LLaVA (1.35-1.44) and 1.60 in
   Gemma (1.57-1.62). Since d' = gap/SD is an identity, the empirical content is
   the near-constancy of the spread contraction: given only the geometric gap
   ratio, one predicts the rank-sensitivity loss with a single per-model gain.

   Out-of-sample check (`pilot/law_validation.py`, `results/law_validation.json`):
   leave-one-cell-out MAE 0.056, against 0.179 for a unit slope and 0.202 for the
   training mean; per-model slopes give RMSE 0.011 against 0.060 for one shared
   slope. A scene bootstrap re-estimating the no-hint axis in every draw puts the
   through-origin slope at 1.423 [1.369,1.473] (500 draws). Leave-one-model-out
   works for 7B (0.018) and LLaVA (0.017) but misses 3B (0.119) and Gemma
   (0.116), the two extreme gains: the per-model gain must be measured inside a
   new model, it does not extrapolate. Phi-3.5-vision, run after the fit, gives
   a through-origin slope of 1.13 and RMSE 0.096 against the fitted 1.425
   (`results/phi_extra.json`), below the fitted 1.25-1.60 range. VOC 2012, a
   different stimulus distribution, gives 1.20 and RMSE 0.123
   (`results/voc_extra.json`).

4. **Behavioural transfer of the dilution law.** Replacing the probe d' with a
   behavioural d' computed from outputs alone (the logit of p(unknown) over
   present and absent items) removes the algebraic identity, because the output
   spread is free to vary. Over the eight hinted cells (4 models x 2
   conflicting hues; the correct-hint cell has no hinted absent arm),
   d'_beh ratio = 0.97 gap ratio + 0.12 with R^2 = 0.57, through-origin 1.27
   and scene-bootstrap slope [0.72, 1.24]; an argmax-rate d' gives 1.47
   (R^2 = 0.51) (`results/law_behaviour.json`). The gain is model-specific:
   0.92 in 3B, 1.43 in 7B, 1.62 in LLaVA and 1.40 in Gemma. The dilution law
   therefore transfers to the model's own output signal in direction and order
   of magnitude, and the residual factor is the output-head term that item 5
   describes.

   Scope: the behavioural fit is descriptive, not predictive. Leave-one-cell-out
   MAE is 0.100 against 0.125 (training mean) and 0.121 (unit slope); leaving a
   model out misses its cells by 0.21 (3B), 0.02 (7B), 0.07 (LLaVA) and 0.25
   (Gemma). What transfers is the linear form and an order-of-magnitude scale;
   with two hinted cells per model the constant is fitted, not predicted, so the
   paper must not present this as a cross-model law.

5. **What is not supported.** The model's own confidence collapse is *not*
   fully predicted by this geometry: Gemma has the same dilution and rotation
   as the Qwen models (0.529, 28.2 deg) but p(unknown) barely degrades (0.961
   to 0.891) and max probability rises (0.627 to 0.701), while Phi-3.5 rotates
   furthest (86°/74°) and its own signal falls below chance (0.939 to
   0.385/0.458). The dissociation is therefore between the representation
   (shared geometry, with a simple dilution-to-d' gain) and the output head
   (model-specific sensitivity to the hint). The behavioural gain ranges
   0.92-1.62, so this is not a universal constant.

Consequence for positioning: the defensible novelty is two-part---the
controlled 2x3 design that isolates the assertion, and the dissociation between
a shared evidence geometry (which carries a quantitative dilution-to-d'
regularity that transfers to outputs) and model-specific output collapse. It is
not a universal law from geometry to behaviour: the transfer holds with a
model-specific gain of 0.92-1.62, which is itself a measurable property of the
output head rather than a constant.

## 12. Cross-backbone control battery (Gemma, Phi, VOC)

Data availability: the per-analysis files for this battery (`control_causal_{gemma,phi}_final.json`, `batch44_{gemma,phi}.json`, `extra44_{gemma,phi}_stats.json`, `control_battery_status.log`, `post_battery_status.log`) were produced on the rented ModelScope A10 and read during analysis, but not copied back before the instance was stopped. They persist on the `/mnt/workspace` volume and can be pulled at the next start; every number used in the paper and in this section is transcribed below. The earlier raw runs for these models (patch, steer, conflict, matched) are local under `results/runs/`.

This batch closes the controls that were missing outside the Qwen family and on
the second stimulus set. All patching numbers below use the same-image source
(the image's own no-hint state) pasted into the conflict pass at the stated
layer; recovery is (patched refusal - base) / (no-hint refusal - base), with
argmax refusal as the metric.

### 12a. Layer-wise patching

| model | layers | base | no hint | same-source recovery | shuffled recovery |
|---|---|---|---|---|---|
| Gemma-3-4B (34) | L8 (24%) | 0.402 | 0.843 | 0.009 | --- |
| | L16 (47%) | | | 0.122 [0.054, 0.192] | -0.012 |
| | L24 (71%) | | | 0.896 [0.822, 0.962] | 1.035 |
| | L30 (88%) | | | 1.009 [1.000, 1.029] | 1.035 |
| Phi-3.5-vision (32) | L14 (44%) | 0.322 | 0.897 | 0.307 [0.221, 0.388] | -0.019 |
| | L22 (69%) | | | 0.973 [0.933, 1.013] | 1.067 |
| | L28 (88%) | | | 1.020 [1.000, 1.046] | 1.050 |
| | L30 (94%) | | | 1.007 [1.000, 1.022] | 1.020 |
| VOC-3B (36) | L16 (44%) | 0.644 | 0.894 | -0.244 [-0.47, -0.11] | -0.276 |
| | L24 (67%) | | | 0.289 [0.11, 0.44] | 0.238 |
| | L30 (83%) | | | 0.844 [0.71, 0.97] | 0.975 |
| | L34 (94%) | | | 1.000 [1.00, 1.00] | 1.020 |

Same-image present-question state (drives refusal below base): Gemma L24
-0.704, L30 -0.736; Phi L22 -0.267, L30 -0.333. Paired same-absent minus
same-present contrasts: Gemma +0.705/+0.770 (L24/L30), Phi +0.713/+0.770
(L22/L30). Same-minus-shuffled: Gemma -0.062 [-0.127, 0.004] (L24), Phi -0.054
[-0.104, -0.004] (L22). Shuffled sources recover as much or more, as in the
Qwen-family models.

No-op / zero paste (patch6): argmax agreement with the no-op is 1.000 and the
maximum probability change is 0 at both probed layers in both models; a zero
paste keeps the refusal decision in every item (change 0.0) but changes the
argmax in 90.0%/80.5% of absent items in Gemma (L24/L30) and 76.6%/91.6% in
Phi (L22/L30).

### 12b. Category-matched control

| model | no hint | frozen | refit | p(unknown) | frozen - p(unknown) | geometry |
|---|---|---|---|---|---|---|
| Gemma | 0.947 | 0.887 | 0.933 | 0.896 | -0.009 [-0.032, 0.011] | 0.546 / 23.8 deg / 0.500 |
| Phi | 0.961 | 0.592 | 0.950 | 0.446 | +0.145 [0.063, 0.232] | 0.547 / 81.5 deg / 0.081 |

Gemma replicates the main-pair result: within a fixed question the frozen probe
does not beat the model's own signal. Phi is the exception, but its own signal
falls below chance (0.446) and the probe itself is only 0.592 against a 0.950
refit, so the margin is "least bad", not good. Within-question gate at 25%
answered coverage (fabrication, probe / p(known) / max prob): Gemma
0.131 / 0.115 / 0.254; Phi 0.438 / 0.446 / 0.592.

### 12c. Deployment and conformal risk control

Matched answered coverage (hint a, scene-cross-fitted probe; probe / p(unknown)
/ max prob):
- 25% coverage: Gemma 0.015 / 0.138 / 0.254; Phi 0.285 / 0.562 / 0.662.
- 50% coverage: Gemma 0.080 / 0.172 / 0.337; Phi 0.556 / 0.586 / 0.636
  (Phi must select 82% of items to answer half).
- Paired scene bootstrap of probe minus p(unknown) at 25% (1000 draws,
  `pilot/gate_ci.py`): Gemma -0.123 [-0.185, -0.069]; Phi -0.277
  [-0.354, -0.192]. At 50%: Gemma -0.092 [-0.126, -0.061]; Phi -0.034
  [-0.073, -0.004]. All intervals exclude zero.

Prevalence sweep at the 25% threshold (probe, p(unknown)): Gemma 0.001 (0.014),
0.003 (0.029), 0.009 (0.082), 0.025 (0.211) at 5/10/25/50%; Phi 0.014 (0.043),
0.029 (0.087), 0.081 (0.222), 0.210 (0.461).

Conformal (130 calibration scenes, answered subset in numerator and
denominator, Clopper-Pearson 90% bound, 50-point scan, family-wise
Bonferroni): Gemma infeasible at alpha=5%, feasible in 4/20 splits at alpha=10%
(mean test risk 0.057 at answered coverage 0.295; minimum UCB 0.119); Phi
infeasible at both (minimum UCB 0.305). Pointwise alpha=10%: Gemma 20/20
splits, mean held-out risk 0.047 at coverage 0.354; Phi 0/20.

### 12d. Conflict steering (readout-level sufficiency)

Same sweep as the Qwen-family models (`steer_conflict_*.jsonl`: base, ablate,
a±0.25, a±0.50 applied during the conflicting pass), analysed with
`pilot/analyze_steer_conflict.py`. Means are over 261 scenes; CIs are paired
scene bootstraps.

| model | setting | absent refusal | present refusal | absent hint | present gold |
|---|---|---|---|---|---|
| Gemma-3-4B | base | 0.439 | 0.008 | 0.540 | 0.241 |
| | a+0.25 | -0.050 [-0.071, -0.033] | -0.004 [-0.011, 0.000] | +0.048 [+0.029, +0.069] | +0.004 [-0.019, +0.023] |
| | a-0.25 | +0.034 [+0.015, +0.054] | 0.000 [0.000, 0.000] | -0.034 [-0.054, -0.015] | -0.004 [-0.023, +0.015] |
| | a+0.50 | -0.109 [-0.140, -0.082] | -0.008 [-0.019, 0.000] | +0.105 [+0.080, +0.130] | +0.011 [-0.011, +0.038] |
| | a-0.50 | +0.056 [+0.029, +0.082] | 0.000 [0.000, 0.000] | -0.061 [-0.088, -0.034] | -0.019 [-0.046, +0.008] |
| Phi-3.5-vision | base | 0.374 | 0.548 | 0.615 | 0.080 |
| | a+0.25 | -0.201 [-0.239, -0.167] | -0.169 [-0.215, -0.126] | +0.169 [+0.138, +0.201] | +0.080 [+0.050, +0.111] |
| | a-0.25 | +0.239 [+0.201, +0.278] | +0.119 [+0.080, +0.161] | -0.178 [-0.213, -0.145] | -0.080 [-0.111, -0.050] |
| | a+0.50 | -0.330 [-0.374, -0.284] | -0.349 [-0.406, -0.287] | +0.226 [+0.187, +0.264] | +0.199 [+0.153, +0.245] |
| | a-0.50 | +0.435 [+0.389, +0.481] | +0.222 [+0.172, +0.276] | -0.238 [-0.278, -0.195] | -0.203 [-0.249, -0.157] |

Gemma's absent gate responds monotonically but with a much smaller amplitude
than the Qwen-family models: at +/-0.50 the shift is +0.056/-0.109, against
0.435/-0.330 in Phi. Its present arm is already at the 0.008 refusal floor, so
the downward direction has no room. Phi is the opposite extreme: conflict
leaves it refusing 55% of present items, and the same direction moves both
arms (present refusal -0.349 at +0.50), so the gate there is not specific to
the absent question. The direction is sufficient at the readout in all five
models, but its amplitude is model-specific.

The no-hint reference for Gemma is absent refusal 0.843 / present refusal
0.050; for Phi 0.897 / 0.119.

### 12e. Site and direction controls (Gemma, Phi, conflict pass)

Same battery as 4d, with the direction fitted on each model's no-hint states
and injected during the conflicting pass. Layer doses are 0.25 of that layer's
median last-token norm; head rows inject at the unembedding input and also carry
three random unit directions of the same norm orthogonal to `w`, plus
ablations. All entries are paired differences from the unsteered conflict base;
CIs are paired scene bootstraps with 2000 draws (`control_causal_gemma_final.json`,
`control_causal_phi_final.json`).

Gemma-3-4B (34 layers; base absent refusal 0.439, present refusal 0.008,
present gold 0.241, present hint 0.674, absent hint 0.540):

| site | absent refusal -/+ | present gold -/+ | present hint -/+ | absent hint -/+ |
|---|---|---|---|---|
| L8 (24%) | -0.146 [-0.174,-0.119] / -0.103 [-0.136,-0.075] | +0.042 / -0.080 | -0.050 / +0.088 | +0.142 / +0.111 |
| L16 (47%) | +0.280 [0.230,0.330] / -0.339 [-0.389,-0.293] | -0.238 / -0.149 | -0.674 / -0.540 | -0.494 / -0.111 |
| L24 (71%) | +0.010 [-0.015,0.034] / -0.092 [-0.125,-0.061] | -0.142 / -0.027 | -0.375 / -0.080 | -0.262 / +0.065 |
| L30 (88%) | +0.088 [0.059,0.117] / -0.151 [-0.184,-0.121] | -0.023 / +0.004 | -0.015 / -0.008 | -0.151 / +0.096 |

Head: absent refusal moves +0.034 [0.015,0.054] / -0.050 [-0.071,-0.033] at
+/-0.25 and +0.056 [0.029,0.082] / -0.109 [-0.140,-0.082] at +/-0.50, with
present gold and hint unchanged (all |delta| <= 0.019, CIs include 0). But the
three random directions move absent refusal by a mean absolute 0.076 against
0.042 for `w` (ratio 0.55; absent hint 0.075 vs 0.041, ratio 0.55), and
ablating `w` is null (-0.002 [-0.006,0.000]), no larger than ablating a random
direction (<=0.004) although `w` removes 5.9x the energy. Gemma's head is
movable but not direction-specific. Among layer sites only L30 is clean: both
signs move the absent gate and present gold/hint are unchanged (|delta| <=
0.023, CIs include 0). L16 and L24 at the negative sign move the gate but
destroy the present arm (gold and hint adoption fall by 0.14-0.67), and L8
lowers absent refusal at both signs. Patching still recovers 90% at L24 and
101% at L30, so the gate is written late in Gemma; what fails is the claim that
`w` is a privileged readout direction.

Phi-3.5-vision (32 layers; base absent refusal 0.374, present refusal 0.548,
present gold 0.080, present hint 0.360, absent hint 0.615):

| site | absent refusal -/+ | present gold -/+ | present hint -/+ | absent hint -/+ |
|---|---|---|---|---|
| L14 (44%) | +0.180 [0.148,0.215] / -0.193 [-0.232,-0.159] | -0.038 / +0.100 | -0.065 / +0.100 | -0.176 / +0.190 |
| L22 (69%) | +0.057 [0.034,0.080] / -0.111 [-0.140,-0.082] | -0.027 / +0.042 | -0.042 / +0.046 | -0.057 / +0.113 |
| L28 (88%) | +0.036 [0.011,0.059] / -0.136 [-0.167,-0.107] | 0.000 / +0.042 | -0.034 / +0.042 | -0.038 / +0.142 |
| L30 (94%) | +0.103 [0.071,0.138] / -0.128 [-0.163,-0.096] | +0.050 / +0.046 | -0.031 / -0.008 | -0.109 / +0.130 |

Head: absent refusal moves +0.239 [0.201,0.278] / -0.201 [-0.239,-0.167] at
+/-0.25 and +0.435 [0.389,0.481] / -0.330 [-0.374,-0.284] at +/-0.50; `w`'s
mean absolute effect is 0.220 against 0.073 for random directions (ratio 3.04),
and ablating `w` is 12.9x a random ablation. The same direction also moves
present refusal (-0.207/+0.107 at L14; -0.349 at head +0.50) and present gold
(+0.100/-0.038 at L14; +0.195/-0.050 at head +0.50), i.e. it controls
answering as a whole rather than an absent-specific gate. Layer action is
largest at L14 (44% depth) and weakens deeper, while patching recovers only 31%
at L14 and 97% at L22: the axis is available to the readout early, but the
conflict-induced loss is localised late. Injection establishes sufficiency of
the direction, not its computation site.

Limitation: random-direction controls exist only at the head, in every model;
layer-site specificity is untested and those profiles are descriptive.

### 12f. Energy-matched removal (Gemma, Phi)

Same design as the Qwen-family energy matching (`pilot/extra44.py --lite`): each
ablation removes a direction rescaled so that its mean removed energy
(mean |x.v| / median final-state norm) matches the stated target. Entries are
absent refusal under the conflicting pass and paired differences from base
(95% scene bootstrap, 2000 draws). `w->r` scales `w` down to a random
direction's energy; `r->w` scales a random direction up to `w`'s energy;
`cm`/`pc1` are the class-mean and first-PC directions.

| model | base | w raw | w->r | r->w | cm->w | pc1->w |
|---|---|---|---|---|---|---|
| Gemma-3-4B | 0.439 | 0.437 (-0.002 [-0.006,0.000]) | 0.437 (-0.002 [-0.006,0.000]) | 0.437 (-0.002 [-0.010,0.004]) | 0.460 (+0.021 [+0.010,+0.033]) | 0.460 (+0.021 [+0.010,+0.033]) |
| Phi-3.5-vision | 0.374 | 0.316 (-0.057 [-0.079,-0.036]) | 0.360 (-0.013 [-0.025,-0.004]) | 0.421 (+0.048 [+0.029,+0.069]) | 0.216 (-0.157 [-0.193,-0.125]) | 0.125 (-0.249 [-0.291,-0.211]) |

`w` carries 5.9x (Gemma) and 7.3x (Phi) the mean projection energy of a random
direction. In Phi the fractions are 0.080 for `w`, 0.011-0.020 for the random
directions, and 0.148/0.145 for the class-mean and first-PC directions, so the
class directions are scaled *down* to `w`'s energy and still move refusal
several times more.

Reading. In Gemma there is no removal effect to attribute: raw and both
matched variants of `w` are null, while class-mean and first-PC directions at
`w`'s energy raise refusal by 0.021 each. The head readout depends on class
structure, not on `w`. In Phi the raw effect is mostly removed energy: matching
`w` down shrinks it from -0.057 to -0.013, and a random direction scaled up to
`w`'s energy moves refusal more and in the opposite direction (+0.048).
Removing class-mean and first-PC directions at matched energy is far stronger
(-0.157, -0.249), so what is causally load-bearing at the readout is the class
subspace, not the evidence direction. Across all five models the residual
matched effect of `w` is at most 0.073, and in 7B, LLaVA and Phi a matched
random direction moves refusal at least as much (in Phi, more and opposite).
Injection efficiency and removal relevance dissociate: the direction that
steers the gate best is not the direction whose removal matters most.

Dose-response is not an artifact of a fixed absolute norm: rescaling the dose
to each prompt's own final-state norm gives absent-refusal changes
+0.050/-0.080 at -/+0.50 in Gemma and +0.464/-0.287 in Phi, preserving the
head-injection sign pattern.

### 12g. What this battery does not establish

Self-critique after completing the battery. None of these gaps is repairable
from the data already on disk; the first is the main one, and it is the reason
the paper claims readout-level sufficiency rather than a circuit.

1. **Layer-site random controls were never run.** Random orthogonal directions
   exist only at the head, in all five models. The layer profiles (L8-L30) mix
   "site matters" with "any same-norm perturbation at that site matters". Gemma
   L30 is clean for `w`, but no random direction was injected there. The layer
   results are therefore descriptive; the causal claim is restricted to the
   readout, where the random controls do exist.
2. **The causal direction is fit in-sample.** `w` is estimated on all no-hint
   states and its injection/ablation is evaluated on the same scenes. The probe
   transfer results are cross-fitted; the causal ones are not. A split-half fit
   would test direction stability and was not run.
3. **Direction and energy are confounded by construction.** Mean-energy
   matching uses the evaluation scenes and a scalar rescaling; per-item energy
   matching would be stricter. Even matched, ablation removes a one-dimensional
   projection, not a mechanism.
4. **The random controls are three directions from one seed.** CIs are
   pointwise over a grid of sites, doses and models, with no multiplicity
   correction across the grid; they are descriptive, not family-wise.
5. **The category-matched control weakens the deployment story.** The frozen
   margin over `p(unknown)` is +0.123/+0.142/+0.142 on the main pair but
   +0.050/+0.002/-0.015 within a fixed question, and the 25%-coverage
   fabrication advantage shrinks, vanishes in 7B and reverses in LLaVA. Part of
   the main-design transfer is arm-specific. This is reported in the paper, not
   hidden, and the refit probe plus the geometry are the parts that survive.
6. **Conformal risk control is underpowered.** At alpha=10% a family-wise
   certificate exists only for Gemma (4/20 splits); minimum bounds are
   0.174/0.210/0.191/0.119/0.305. No deployment guarantee is claimed.
7. **Labels are single-annotator.** COCO/VOC annotations plus an automated
   canonical-colour mapping; no inter-annotator agreement was collected.
   Residual colour-label noise is not quantified, though probe AUROCs near 0.99
   bound its plausible size.
8. **External validity is task-specific.** Five models from three language
   families and two stimulus sets, but all questions are attribute
   classification; the counting and left-right forms exist only for the Qwen
   family. VizWiz is the only natural unanswerable set, and there the probe
   does not beat `p(unknown)`.
9. **No mechanism or training origin.** The work identifies a decodable
   direction, a late gate and a dilution/rotation law; it does not identify the
   circuit that computes either, nor why training produces this geometry.

Decision: no further GPU run is justified for the claims as written. The one
experiment that would strengthen a stronger claim, layer-site random-direction
controls at one deep layer per model, tests a claim the paper deliberately does
not make; it is left as future work rather than run to inflate the appendix.
