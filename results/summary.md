# Result digests

These are the numbers reported in the paper, transcribed from the stdout of the
analysis scripts on a single 24 GB A10. The scripts are deterministic given the
frozen inputs (fixed seeds, grouped CV, bootstrap with fixed resamples); rerun
them to regenerate the underlying `.jsonl` / `.npz` files and full logs.

| Run | Script | Raw stdout (remote) |
|---|---|---|
| Main 2x3 behaviour | `pilot/analyze_v5.py` | `runs/repro/main_3b.txt` |
| Evidence probe | `pilot/probe.py` | `runs/repro/probe_3b.txt` |
| Output-space baselines | `pilot/baselines.py` | `runs/repro/baselines_3b.txt` |
| Selective gate | `pilot/selective.py` | `runs/repro/selective_3b.txt` |
| Steering / held-out half | `pilot/steer.py`, `pilot/steer_half.py` | `runs/repro/steer_3b.txt`, `steer_half.txt` |
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
| final-state probe | 0.995 [0.989, 0.999] | 0.998 [0.994, 1.000] | 0.997 [0.994, 0.999] |
| p(unknown) | 0.993 | 0.834 | 0.969 |
| max probability | 0.615 | 0.463 | 0.916 |
| margin (colour max - unknown) | 0.985 | 0.766 | 0.959 |
| negative entropy | 0.640 | 0.364 | 0.902 |
| top-2 gap | 0.934 | 0.564 | 0.927 |
| 12-d logit probe | 0.982 | 0.923 | 0.974 |

Probe minus p(unknown): no hint +0.002 [-0.003, +0.007]; conflict +0.164
[+0.132, +0.196]; correct +0.028 [+0.017, +0.042].

Transfer, fitted on the no-hint pair and evaluated on the hinted pair:
none -> conflict 0.969; none -> correct 0.996. Reverse transfer (conflict ->
none) 0.999.

## 3. Controls

| control | probe, no hint | probe, conflict | p(unknown), no hint | p(unknown), conflict | maxprob, no hint | maxprob, conflict |
|---|---|---|---|---|---|---|
| main pair (grouped by sample) | 0.995 | 0.998 | 0.993 | 0.834 | 0.615 | 0.463 |
| grouped by category | 0.995 | 0.997 | — | — | — | — |
| category-matched pairs, grouped by category | 0.985 | 0.973 | 0.983 | 0.817 | 0.701 | 0.468 |
| gray image (text only) | 0.849 | 0.981 | 0.700 | 0.325 | — | — |

The direction learned on gray images transfers across hints at 0.540 / 0.603,
versus 0.969 / 0.996 for the real-image direction.

## 4. Steering (3B)

In-sample direction, n=261:

| setting | present refusal | present top-1 conf. | absent refusal | absent adopts hint |
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
| LLaVA-OneVision-7B (28 layers) | no hint | 0.751 | 0.579 | 100% |
| | conflict | 0.161 | 0.194 | 0% |
| | same L18 (64%) | 0.314 | 0.299 | 27% |
| | same L24 (86%) | 0.648 | 0.480 | 74% |
| | same L26 (93%) | 0.743 | 0.550 | 93% |
| | shuffled L26 | 0.781 | 0.585 | 102% |

## 6. Selective gate: fabrication at fixed coverage under conflict

| coverage | 3B probe | 3B p(unknown) | 3B maxprob | 7B probe | 7B p(unknown) | 7B maxprob | LLaVA probe | LLaVA p(unknown) | LLaVA maxprob |
|---|---|---|---|---|---|---|---|---|---|
| 50% | 0.011 | 0.249 | 0.510 | 0.019 | 0.268 | 0.648 | 0.019 | 0.295 | 0.655 |
| 70% | 0.398 | 0.456 | 0.762 | 0.398 | 0.521 | 0.835 | 0.398 | 0.525 | 0.812 |
| 90% | 0.801 | 0.801 | 0.985 | 0.801 | 0.839 | 0.969 | 0.801 | 0.816 | 0.969 |

At 70% coverage the probe sits on the structural floor: a half-absent set
answered at 70% coverage must contain at least 40% absent items.

## 7. Task-form boundaries (3B)

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

| model | probe, no hint | probe, conflict | p(unknown), no hint | p(unknown), conflict | maxprob, conflict | 12-d logit probe, conflict |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 0.995 | 0.998 | 0.993 | 0.834 | 0.463 | 0.923 |
| Qwen2.5-VL-7B | 0.995 | 0.998 | 0.984 | 0.797 | 0.306 | 0.896 |
| LLaVA-OneVision-7B | 0.996 | 0.999 | 0.991 | 0.781 | 0.289 | 0.909 |
