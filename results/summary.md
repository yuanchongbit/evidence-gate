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

The paper's controls (gray images, category-matched pairs, category-grouped CV)
come from `pilot/blank.py` and `pilot/matched.py`; their key numbers are repeated
below.

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
| frozen final-state probe (headline) | 0.995 [0.989, 0.999] | 0.958 [0.940, 0.973] | 0.995 [0.987, 0.999] |
| refit final-state probe (upper bound) | 0.995 [0.989, 0.999] | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] |
| p(unknown) | 0.993 | 0.834 | 0.969 |
| max probability | 0.615 | 0.463 | 0.916 |
| margin (colour max - unknown) | 0.985 | 0.766 | 0.959 |
| negative entropy | 0.640 | 0.364 | 0.902 |
| top-2 gap | 0.934 | 0.564 | 0.927 |
| 12-d logit probe | 0.982 | 0.923 | 0.974 |

Probe minus p(unknown): no hint +0.002 [-0.003, +0.007]; conflict +0.164
[+0.132, +0.196]; correct +0.028 [+0.017, +0.042].

Transfer, fitted on the no-hint pair and evaluated on the hinted pair. The
frozen row above is this transfer; refitting inside the condition is an upper
bound. Frozen transfer to conflict is 0.958 for hint b and 0.957 for hint a;
to correct 0.995. LLaVA 0.923, Qwen-7B 0.947 (hint b). Reverse transfer
(conflict -> none) is 0.999. Headline numbers in the paper are frozen.

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
direction still ranks conflict pairs at AUROC 0.958; a direction refit on the
conflict pair reaches 0.998. Reading: the hint compresses the evidence axis;
ranking survives, scale does not. The residual 22% gain loss is real but
modest. The criterion shift is 3B-specific and inconsistent across models.
Figure: paper/figures/fig_sdt.pdf.

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


## 3. Controls

| control | refit probe, no hint | refit probe, conflict | p(unknown), no hint | p(unknown), conflict | maxprob, no hint | maxprob, conflict |
|---|---|---|---|---|---|---|
| main pair (grouped by sample) | 0.995 | 0.998 | 0.993 | 0.834 | 0.615 | 0.463 |
| grouped by category | 0.995 | 0.997 | — | — | — | — |
| category-matched pairs, grouped by category | 0.985 | 0.973 | 0.983 | 0.817 | 0.701 | 0.468 |
| gray image (text only) | 0.849 | 0.981 | 0.700 | 0.325 | — | — |

The direction learned on gray images transfers across hints at 0.540 / 0.603,
versus 0.958 / 0.995 for the frozen real-image direction (0.969 / 0.996 when
refitted on the condition).

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

## 6. Selective gate: fabrication at fixed coverage under conflict

Deployment numbers use the frozen probe (fitted on the no-hint pair) and the
plausible hint pair (`sufficient|wrong` vs `absent|a`). The refit gate is an
upper bound and is not deployable.

| coverage | 3B frozen | 3B refit | 3B p(unknown) | 3B maxprob | 7B frozen | 7B p(unknown) | LLaVA frozen | LLaVA p(unknown) |
|---|---|---|---|---|---|---|---|---|
| 50% | 0.069 | 0.011 | 0.249 | 0.510 | 0.126 | 0.268 | 0.111 | 0.295 |
| 70% | 0.398 | — | 0.456 | 0.762 | 0.398 | 0.521 | 0.398 | 0.525 |
| 90% | 0.801 | — | 0.801 | 0.985 | 0.801 | 0.839 | 0.801 | 0.816 |

At 70% coverage the probe sits on the structural floor: a half-absent set
answered at 70% coverage must contain at least 40% absent items. The frozen
gate reduces fabrication by 3.6x / 2.1x / 2.7x at 50% coverage. A logistic
readout over the 12 output probabilities (3B, refit) fabricates 0.142, so the
gain is not just a learned function of the logits.

Prevalence shift, 3B threshold calibrated on the balanced set, fabrication
among answered (probe vs p(unknown)):

| unanswerable prevalence | 5% | 10% | 25% | 50% |
|---|---|---|---|---|
| 3B probe (p(unknown)) | 0.007 (0.032) | 0.016 (0.066) | 0.045 (0.174) | 0.125 (0.387) |
| 7B probe (p(unknown)) | 0.028 (0.050) | 0.058 (0.100) | 0.155 (0.250) | 0.355 (0.500) |
| LLaVA probe (p(unknown)) | 0.007 (0.023) | 0.014 (0.048) | 0.042 (0.132) | 0.116 (0.314) |

Split conformal risk control, 130 calibration scenes, Clopper-Pearson 90% UCB
over all prefixes, evaluated on the held-out half:

| model | alpha=5% | alpha=10% | coverage | test fabrication |
|---|---|---|---|---|
| 3B | infeasible (UCB 0.061) | certified | 0.519 | 0.088 |
| 7B | certified on calibration | — | 0.340 | 0.067 |
| LLaVA | infeasible (UCB 0.059) | certified | 0.511 | 0.127 |

The 7B 5% certificate does not transfer to the held-out half. A pollution
detector `probe - p(unknown)` is not useful: present-vs-absent AUROC is
0.97 / 0.95 / 0.96, but at 50% coverage it selects the same items as the probe
alone (fabrication 0.069 / 0.119 / 0.111).

Always-on and fixed-threshold checks (3B). Pooled over all three hint
conditions, ranking by the no-hint probe gives 50%-coverage fabrication 0.064
versus 0.193 for p(known)=1-p(unknown); at 70% coverage, 0.400 versus 0.480.
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
| Qwen2.5-VL-3B | 0.995 | 0.958 | 0.998 | 0.993 | 0.834 | 0.463 | 0.923 | 0.463 [0.438, 0.488] |
| Qwen2.5-VL-7B | 0.995 | 0.947 | 0.998 | 0.984 | 0.797 | 0.306 | 0.896 | 0.424 [0.393, 0.457] |
| LLaVA-OneVision-7B | 0.996 | 0.923 | 0.999 | 0.991 | 0.781 | 0.289 | 0.909 | 0.344 [0.324, 0.365] |

The frozen column is the deployable headline; the refit column is an upper
bound from retraining inside the conflict condition.

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

Removing the abstention token removes abstention by construction and raises
hint adoption; a sentence-form hint is not treated as an assertion and the
conflict effect disappears. LLaVA moves the same way from a lower baseline
(conflict refusal 0.088 -> 0.559, adoption 0.904 -> 0.333, wrong-hint accuracy
0.360 -> 0.548). Both are stated as scope conditions, not as robustness checks
that pass.

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
