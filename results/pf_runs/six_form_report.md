# Six prompt forms: is the conflict effect an artifact of the bracketed hint?

Run: 2026-11-07/08, Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct and
Gemma-3-4B-it, same 261 scenes, final-token states, CPU probes. The question
is the external reviewer's: who moves the evidence axis, and is it the
bracketed `Hint:` prefix rather than the assertion?

Design: six forms, each run in the cells `absent|none`, `absent|a`,
`absent|b`, `sufficient|none`, `sufficient|wrong`, plus the agreeing-hint
cell `sufficient|correct` in the new battery.

| form | assertion placement |
|---|---|
| `std` | `Hint: the cat is red.` before the question, user message |
| `nl` | `A colleague mentioned that the cat is red.` before the question |
| `claim` | `The cat is red.` before the question |
| `belief` | `I think the cat is red.` before the question |
| `system` | `The cat is red.` appended to the system message |
| `dialogue` | user turn `I think the cat is red.`, assistant `Okay.`, then image + question; the no-hint cell uses a neutral first turn |

The `std` form reproduces the main battery cell for cell in both models
(3B: refusal 0.904/0.521/0.563, adoption 0.467/0.433, wrong-hint accuracy
0.383; 7B: 0.958/0.877/0.958, 0.123/0.042, 0.249), so the battery is a
faithful re-run and the new forms differ only in the assertion.

## 1. Behaviour is strongly format-dependent

| form | adopt a | adopt b [95% CI] | refuse none | refuse a | refuse b | acc wrong |
|---|---|---|---|---|---|---|
| 3B std | 0.467 | 0.433 [0.372,0.494] | 0.904 | 0.521 | 0.563 | 0.383 |
| 3B nl | 0.142 | 0.042 [0.019,0.069] | 0.904 | 0.854 | 0.950 | 0.567 |
| 3B claim | 0.176 | 0.115 [0.077,0.157] | 0.904 | 0.812 | 0.877 | 0.602 |
| 3B belief | 0.161 | 0.088 [0.054,0.123] | 0.904 | 0.808 | 0.889 | 0.659 |
| 3B system | 0.927 | 0.885 [0.847,0.920] | 0.904 | 0.057 | 0.107 | 0.598 |
| 3B dialogue | 0.605 | 0.483 [0.425,0.540] | 0.939 | 0.383 | 0.510 | 0.644 |
| 7B std | 0.123 | 0.042 [0.019,0.069] | 0.958 | 0.877 | 0.958 | 0.249 |
| 7B nl | 0.004 | 0.000 [0.000,0.000] | 0.958 | 0.989 | 0.996 | 0.487 |
| 7B claim | 0.100 | 0.050 [0.027,0.077] | 0.958 | 0.900 | 0.950 | 0.249 |
| 7B belief | 0.023 | 0.008 [0.000,0.019] | 0.958 | 0.969 | 0.985 | 0.479 |
| 7B system | 0.084 | 0.019 [0.004,0.034] | 0.958 | 0.908 | 0.973 | 0.521 |
| 7B dialogue | 0.011 | 0.000 [0.000,0.000] | 0.977 | 0.981 | 0.989 | 0.506 |

In 3B, adoption of the plausible absent hint spans a factor of 20, from
0.042 (`nl`) to 0.885 (`system`); the bracketed `std` (0.433) sits in the
middle and the two-turn `dialogue` (0.483) is statistically indistinguishable
from it (paired difference -0.050 [-0.123,+0.023]). For this model the
bracketed form is therefore neither necessary for adoption (system, dialogue)
nor the strongest form (system beats it by +0.452 [+0.391,+0.513]).

In 7B adoption is near zero in every form (0.000--0.050), so this model's
readout almost never takes the assertion as the answer, whatever its position.

## 2. Geometry survives every form

| form | length ratio | angle (deg) | cos(hint pair) | std probe transfer |
|---|---|---|---|---|
| 3B std | 0.523 [0.508,0.544] | 38.5 [35.7,41.9] | 0.908 | 0.969 |
| 3B nl | 0.589 [0.574,0.612] | 44.2 [42.2,46.2] | 0.940 | 0.992 |
| 3B claim | 0.595 [0.576,0.618] | 46.9 [44.2,49.7] | 0.949 | 0.985 |
| 3B belief | 0.681 [0.666,0.700] | 32.7 [31.1,34.8] | 0.968 | 0.997 |
| 3B system | 0.672 [0.658,0.691] | 32.5 [31.0,34.2] | 0.919 | 0.999 |
| 3B dialogue | 0.667 [0.653,0.686] | 28.0 [26.9,29.4] | 0.941 | 0.999 |
| 7B std | 0.384 [0.364,0.410] | 47.4 [44.2,51.0] | 0.908 | 0.953 |
| 7B nl | 0.568 [0.554,0.587] | 35.1 [33.9,36.4] | 0.973 | 0.995 |
| 7B claim | 0.393 [0.377,0.414] | 46.9 [44.2,50.0] | 0.914 | 0.958 |
| 7B belief | 0.525 [0.509,0.544] | 39.6 [38.1,41.4] | 0.965 | 0.991 |
| 7B system | 0.672 [0.652,0.693] | 28.0 [26.7,29.5] | 0.970 | 0.998 |
| 7B dialogue | 0.746 [0.729,0.763] | 17.9 [17.3,18.7] | 0.993 | 0.999 |

The evidence axis shortens and turns in every form of both models. The
bracketed form is not special, and in 7B it is the *most* diluted form
(0.384) among the six. A probe fitted only on `std` conflict states transfers
to every other form's conflict states at AUROC 0.953--0.999, so the state
organisation is shared across forms even where behaviour is not.

Caveat: in five forms the two no-hint cells are the same prompt (the
assertion is absent), so their states are identical by construction. The
`v0` axis is therefore the same vector in those forms, and only the
`dialogue` no-hint cell is an independent test (its neutral first turn
changes the prompt; the clean axis is still recovered at cos 0.976 in 3B and
0.960 in 7B against the other forms). All conflict axes use form-specific
prompts, so the length ratios, rotations and transfers are not trivial. In
Gemma the dialogue neutral turn is not axis-preserving (cos 0.689), so its
geometry is read within-form only (Section 7.2).

## 3. Reliability and what the rotation is not

Split-half over scenes (400 bootstrap draws), conflict axis and hint-pair
cosine:

| form | cos(va half1, va half2) | cos(va, vb) | cos(rotation, colour-swap) held out |
|---|---|---|---|
| 3B std | 0.959 [0.936,0.976] | 0.907 [0.881,0.924] | +0.159 [+0.053,+0.313] |
| 3B nl | 0.971 [0.952,0.982] | 0.939 [0.920,0.951] | +0.062 [-0.010,+0.180] |
| 3B claim | 0.969 [0.951,0.980] | 0.948 [0.930,0.959] | +0.017 [-0.065,+0.127] |
| 3B belief | 0.976 [0.962,0.985] | 0.967 [0.957,0.974] | -0.008 [-0.102,+0.108] |
| 3B system | 0.973 [0.957,0.984] | 0.919 [0.894,0.935] | +0.163 [+0.058,+0.291] |
| 3B dialogue | 0.977 [0.962,0.985] | 0.941 [0.924,0.952] | +0.214 [+0.116,+0.354] |
| 7B std | 0.963 [0.945,0.978] | 0.907 [0.870,0.931] | +0.192 [+0.102,+0.289] |
| 7B nl | 0.982 [0.971,0.989] | 0.973 [0.962,0.979] | +0.193 [+0.130,+0.269] |
| 7B claim | 0.966 [0.946,0.981] | 0.914 [0.882,0.936] | +0.260 [+0.181,+0.349] |
| 7B belief | 0.979 [0.967,0.987] | 0.964 [0.952,0.973] | +0.200 [+0.133,+0.264] |
| 7B system | 0.988 [0.982,0.992] | 0.970 [0.960,0.977] | +0.229 [+0.151,+0.318] |
| 7B dialogue | 0.991 [0.987,0.994] | 0.993 [0.990,0.994] | +0.218 [+0.143,+0.306] |

The conflict axis is reliable (0.96--0.99), and the axes measured against
the two different hinted colours are nearly parallel (0.91--0.99); the
rotation is not specific to which colour the assertion names. The rotation is only weakly aligned with the isolated
colour-swap direction measured on held-out scenes (0.16--0.26 in the forms
where it is measurable), so the assertion does not simply add a colour vector
to the evidence axis; it reconfigures the axis. Section 8 shows that this
reconfiguration is not specific to the query either: an added sentence with no
object and no colour turns the axis as far as the conflict axis does in 3B.

## 4. The agreeing-hint control

Behaviour in the sufficient arm; `gain` is accuracy under the correct hint
minus no hint, `loss` is accuracy under the wrong hint minus no hint.

| form | gain correct | loss wrong | len(correct) | len(wrong) | angle correct | angle wrong |
|---|---|---|---|---|---|---|
| 3B std | +0.211 | -0.379 | 0.78 | 0.52 | 39 | 38 |
| 3B nl | +0.069 | -0.195 | 0.75 | 0.59 | 39 | 44 |
| 3B claim | +0.134 | -0.161 | 0.72 | 0.59 | 39 | 47 |
| 3B belief | +0.111 | -0.103 | 0.81 | 0.68 | 30 | 33 |
| 3B system | +0.169 | -0.165 | 0.65 | 0.67 | 29 | 32 |
| 3B dialogue | +0.134 | -0.126 | 0.65 | 0.67 | 27 | 28 |
| 7B std | +0.280 | -0.418 | 0.74 | 0.38 | 22 | 47 |
| 7B nl | -0.015 | -0.180 | 0.75 | 0.57 | 27 | 35 |
| 7B claim | +0.222 | -0.418 | 0.70 | 0.39 | 30 | 47 |
| 7B belief | +0.157 | -0.188 | 0.72 | 0.52 | 30 | 40 |
| 7B system | +0.038 | -0.146 | 0.70 | 0.67 | 24 | 28 |
| 7B dialogue | -0.050 | -0.115 | 0.77 | 0.75 | 17 | 18 |

Two facts matter. First, the behavioural exchange rate is not symmetric: in
9 of 12 model-by-form cells the loss under conflict exceeds the gain under
agreement, and in 7B's `nl` and `dialogue` forms an agreeing assertion does
not help at all. The exceptions are 3B `belief`, `system` and `dialogue`,
where gain and loss are within 0.03 of each other. An assertion is not a
neutral cue that the model weighs symmetrically; conflict is penalised more
than agreement is rewarded.

Second, the geometry separates the two cells: with an agreeing hint the
evidence axis remains longer than with a conflicting one in 4/6 forms in 3B
and 6/6 in 7B, most sharply in 7B `std` (0.74 vs 0.38) and `claim` (0.70 vs
0.39); rotation is present under both and is larger under conflict in 7B. So
dilution is graded by whether the assertion contradicts the image, while
rotation is what any assertion does.

## 5. The stated colour is linearly present in every form

Eleven-way colour decoding from single final-token states (chance 1/11;
majority-class baselines 0.18 in the a cell and 0.38 in the b cell).

| form | 3B within-b CV | 3B a->b raw / balanced | 3B std_a->form_b | 7B within-b CV | 7B a->b raw / balanced | 7B std_a->form_b |
|---|---|---|---|---|---|---|
| std | 0.985 | 0.939 / 0.896 | 0.939 | 0.996 | 0.985 / 0.992 | 0.985 |
| nl | 0.992 | 0.966 / 0.961 | 0.958 | 0.954 | 0.847 / 0.849 | 0.870 |
| claim | 0.985 | 0.900 / 0.856 | 0.912 | 0.989 | 0.962 / 0.968 | 0.950 |
| belief | 0.973 | 0.847 / 0.764 | 0.858 | 0.958 | 0.881 / 0.877 | 0.881 |
| system | 0.996 | 0.962 / 0.962 | 0.962 | 0.992 | 0.958 / 0.962 | 0.954 |
| dialogue | 0.981 | 0.946 / 0.954 | 0.954 | 0.939 | 0.567 / 0.449 | 0.609 |

The a->b test is the important one: the same scene is labelled
`acolor` in the a cell and `DISTRACTOR[acolor]` in the b cell, so a decoder
that used the scene, the category or the image would fail. It succeeds in 11
of 12 cells, far above both the 1/11 chance and the majority baseline. The
exception is the 7B two-turn dialogue (0.57 raw, 0.45 balanced), where the
colour is decodable within the b cell (0.94) but the a->b mapping does not
transfer, i.e. the two turns encode the colour differently. Even there, a
`std`-trained decoder reads the b cell at 0.61.

The conclusion for the reviewer's question: the assertion is represented in
every form, including forms where the model does not adopt it. What the form
changes is whether the readout uses that representation.

## 6. Attention does not explain the format differences

Final-query attention shares in 3B (eager attention, 36 layers, 55 scenes,
images resized to 512 px for this run), with adoption measured on the
battery for the same scenes:

| form | hint a | image a | question a | hint wrong | image wrong | question wrong | adopt a | adopt wrong |
|---|---|---|---|---|---|---|---|---|
| std | 0.062 | 0.071 | 0.204 | 0.055 | 0.089 | 0.190 | 0.49 | 0.43 |
| nl | 0.068 | 0.073 | 0.185 | 0.059 | 0.096 | 0.171 | 0.17 | 0.09 |
| claim | 0.058 | 0.074 | 0.195 | 0.054 | 0.096 | 0.180 | 0.11 | 0.09 |
| belief | 0.063 | 0.076 | 0.189 | 0.054 | 0.099 | 0.172 | 0.17 | 0.07 |
| system | 0.052 | 0.072 | 0.198 | 0.036 | 0.104 | 0.183 | 0.96 | 0.39 |
| dialogue | 0.049 | 0.071 | 0.195 | 0.034 | 0.103 | 0.182 | 0.67 | 0.24 |

The assertion receives only 3--7% of final-query attention in every form.
The form with the highest adoption (`system`, 0.96) has the *lowest*
assertion share (0.052), and the form with the lowest adoption (`nl`, 0.17)
has a higher one (0.068). Image attention rises from the absent to the
sufficient arm in every form (e.g. 0.071 -> 0.089 in `std`, 0.072 -> 0.104 in
`system`), so the readout does look at the image more when it carries
evidence. Within a form, adoption is modestly correlated with assertion
attention across scenes (r = +0.36 `std`, +0.43 `dialogue`, +0.17 `system` in
the absent cell), but the large between-form variation in adoption is not a
matter of how much the readout attends to the assertion. Caveat: this run's
own argmax rates are lower than the battery's because of the 512 px resize,
so the correlations use the battery's adoption, not the attention run's.

## 7. Cross-family replication (Gemma-3-4B)

Run 2026-11-08, same 261 scenes and six forms. The dialogue no-hint cells were
rerun with the neutral first turn and merged before analysis (522/9396 states
replaced). The `std` cell reproduces the main Gemma battery (refusal
0.402/0.475, adoption 0.579/0.502, wrong-hint accuracy 0.241), so the battery
is faithful for this model too.

### 7.1 The behavioural form ordering reverses

| form | adopt a | adopt b [95% CI] | refuse none | refuse a | refuse b | acc wrong |
|---|---|---|---|---|---|---|
| std | 0.579 | 0.502 [0.441,0.563] | 0.843 | 0.402 | 0.475 | 0.241 |
| nl | 0.080 | 0.050 [0.027,0.077] | 0.843 | 0.862 | 0.881 | 0.540 |
| claim | 0.207 | 0.119 [0.084,0.165] | 0.843 | 0.762 | 0.847 | 0.437 |
| belief | 0.057 | 0.038 [0.015,0.065] | 0.843 | 0.843 | 0.851 | 0.533 |
| system | 0.077 | 0.034 [0.015,0.061] | 0.843 | 0.912 | 0.954 | 0.418 |
| dialogue | 0.015 | 0.004 [0.000,0.015] | 0.743 | 0.935 | 0.962 | 0.536 |

This is the mirror image of Qwen 3B. The bracketed `std` form, which sits in
the middle of the Qwen range, is the only Gemma form with substantial
adoption (0.502); `system` and `dialogue`, the two strongest Qwen forms, are
the two weakest in Gemma (0.034 and 0.004). In the dialogue form the
assertion *raises* absent refusal above its own no-hint level (0.743 ->
0.935/0.962), so it backfires. The family x placement interaction is
therefore real: what changes with placement is the readout policy, not the
evidence geometry.

### 7.2 The geometry does not reverse

| form | length ratio [95% CI] | angle (deg) [95% CI] | cos(hint pair) | std probe transfer |
|---|---|---|---|---|
| std | 0.529 [0.500,0.557] | 28.2 [26.4,30.5] | 0.971 | 0.982 |
| nl | 0.885 [0.852,0.918] | 16.3 [15.8,17.1] | 0.998 | 0.997 |
| claim | 0.731 [0.695,0.766] | 25.8 [25.0,27.0] | 0.993 | 0.992 |
| belief | 0.860 [0.831,0.889] | 16.4 [16.0,17.2] | 0.998 | 0.995 |
| system | 0.676 [0.637,0.716] | 31.7 [30.6,33.1] | 0.996 | 0.969 |
| dialogue | 0.891 [0.848,0.936] | 20.9 [20.0,22.5] | 0.999 | 0.990 |

The axis shortens to 0.68--0.89 of its no-hint length and turns 16--32 degrees
in every non-std form, and to 0.529 and 28.2 degrees in std (matching the
earlier `prompt_variants` run: 0.529, 28.2 degrees). The two hinted colours
are nearly parallel (0.971--0.999), split-half reliability is 0.985--0.989,
and a probe fitted on std conflict states transfers to every form at AUROC
0.969--0.997. The rotation is therefore not a property of the bracket: it is
as large or larger in forms that behave very differently.

Caveat: the Gemma dialogue no-hint axis is 46 degrees from the std no-hint
axis (cos 0.689), unlike Qwen (cos 0.960/0.976), so its geometry is
interpretable within-form only. The no-hint refusal rate also differs there
(0.743 vs 0.843), consistent with a template effect of the extra turn.

### 7.3 An agreeing hint dilutes as much as a conflicting one

| form | len(correct) | len(wrong) | angle(correct) | angle(wrong) |
|---|---|---|---|---|
| std | 0.54 | 0.53 | 30 | 28 |
| nl | 0.81 | 0.88 | 19 | 16 |
| claim | 0.71 | 0.73 | 34 | 26 |
| belief | 0.77 | 0.86 | 18 | 16 |
| system | 0.68 | 0.68 | 40 | 32 |
| dialogue | 0.85 | 0.89 | 22 | 21 |

In Qwen the agreeing hint spares part of the dilution (4/6 forms in 3B, 6/6
in 7B). In Gemma it does not: the axis contracts to the same length and
rotates as much (often more) when the assertion is true. The cross-family
invariant is that an assertion moves the axis; whether contradiction adds
movement beyond agreement is family-specific.

### 7.4 Cross-cell colour decoding is form-dependent in Gemma

| form | within-b CV | a->b raw | a->b balanced | baseline b | std_a -> form_b |
|---|---|---|---|---|---|
| std | 0.985 | 0.847 | 0.849 | 0.375 | 0.847 |
| nl | 0.881 | 0.395 | 0.303 | 0.375 | 0.471 |
| claim | 0.962 | 0.659 | 0.628 | 0.375 | 0.648 |
| belief | 0.820 | 0.176 | 0.117 | 0.375 | 0.333 |
| system | 0.908 | 0.464 | 0.408 | 0.375 | 0.479 |
| dialogue | 0.709 | 0.096 | 0.051 | 0.375 | 0.195 |

Within-cell decoding is high in every form (0.71--0.99), but these decoders
can exploit category-level colour priors, because the hinted colour is a
fixed function of the absent category; the a->b test, in which the same
scene carries a different label, is the one that rules this out. That test
succeeds only in `std` (0.85) and `claim` (0.66) and falls to 0.10 in
`dialogue`. In Qwen the same test succeeded in 11/12 cells, including forms
the model ignores. So the Qwen claim "the assertion is represented even when
it is not used" does not carry over unchanged: in Gemma a shared colour code
is demonstrable only in the two forms the model acts on, while the axis moves
in all six. The geometry claim (dilution, rotation, transfer) is the part
that replicates; the cross-cell content code is not.

### 7.5 The geometry-accuracy link replicates; the geometry-adoption link does not

Across the six Gemma forms, the length ratio predicts wrong-hint accuracy at
r = +0.98 (Spearman +0.94), the same sign and at least the same strength as
Qwen 3B (+0.89/+0.83) and 7B (+0.89/+0.93). The same ratio does not have a
consistent relation to adoption: 3B +0.25, 7B -0.79 (adoption near zero
everywhere), Gemma -0.83. Adoption is governed by the refusal policy, which
placement changes differently in each family; conflict accuracy is governed
by how much the assertion dilutes the axis, which is consistent across all
three models.

## 8. Who moves the axis: a query-specificity control

Every form asserts something about the queried category, so the six forms
cannot, on their own, separate query content from the extra sentence or from a
colour word about something else. Three control forms add one sentence to the
`nl` template in both evidence arms, with no hint in either arm:

| control | sentence | object named | colour across arms |
|---|---|---|---|
| `neutral` | "A colleague mentioned that the room is quiet." | none | none |
| `other` | "A colleague mentioned that the remote is {absent_hint}." | a third object | constant |
| `query` | "A colleague mentioned that the {category} is {absent_hint}." | the arm's queried category | constant |

`remote` is one of six COCO categories that occur neither as a queried nor as
an absent category in the manifest, so the sentence cannot name the query. In
the `query` control the asserted colour is the scene's absent-category colour
in both arms and never changes; only the category named follows the arm (the
present category with evidence, the absent category without). All three
controls run the same 261 scenes in 3B, 7B and Gemma. The queried-conflict row
is the `nl` conflict axis from the main battery, reproduced here as a
consistency check (0.589 / 44.2, 0.568 / 35.1 and 0.885 / 16.3, identical to
Sections 2 and 7.2); its arms also change the asserted colour, from the wrong
colour to the absent-category colour.

| model | added material | length ratio [95% CI] | angle to v0 (deg) [95% CI] |
|---|---|---|---|
| 3B | neutral extra sentence | 0.822 [0.803,0.840] | 42.7 [41.4,44.2] |
| 3B | other-object assertion | 0.804 [0.787,0.822] | 31.1 [30.1,32.1] |
| 3B | queried assertion, constant colour | 0.604 [0.587,0.622] | 38.5 [36.6,40.4] |
| 3B | queried assertion (`nl` conflict) | 0.589 | 44.2 |
| 7B | neutral extra sentence | 0.730 [0.711,0.749] | 25.0 [23.9,26.1] |
| 7B | other-object assertion | 0.691 [0.673,0.709] | 28.1 [27.2,29.1] |
| 7B | queried assertion, constant colour | 0.596 [0.577,0.615] | 30.9 [29.4,32.3] |
| 7B | queried assertion (`nl` conflict) | 0.568 | 35.1 |
| Gemma | neutral extra sentence | 1.000 [0.975,1.027] | 10.1 [9.5,10.7] |
| Gemma | other-object assertion | 0.897 [0.869,0.928] | 14.1 [13.6,14.7] |
| Gemma | queried assertion, constant colour | 0.877 [0.842,0.916] | 15.2 [14.7,15.7] |
| Gemma | queried assertion (`nl` conflict) | 0.885 | 16.3 |

Cosine with the conflict axis, neutral / other / query: 3B 0.795 / 0.865 /
0.976; 7B 0.874 / 0.921 / 0.988; Gemma 0.964 / 0.978 / 0.997. Among
themselves the three control axes are close: 3B 0.825--0.908, 7B
0.898--0.945, Gemma 0.967--0.982.

Paired contrast, query minus other (the object the sentence is about changes,
the colour and the template do not): angle +7.37 deg [+5.78,+8.96] and length
-0.20 [-0.22,-0.18] in 3B; +2.76 deg [+1.52,+3.95] and -0.10 [-0.11,-0.08] in
7B; +1.04 deg [+0.52,+1.56] and -0.02 [-0.05,+0.01] in Gemma. Query minus
neutral: 3B -4.25 deg [-6.25,-2.14] and -0.22 [-0.24,-0.20]; 7B +5.89 deg
[+4.17,+7.63] and -0.13 [-0.16,-0.11]; Gemma +5.10 deg [+4.45,+5.75] and
-0.12 [-0.16,-0.09].

| model | control | refusal (absent) | gold acc (sufficient) | answers stated colour (sufficient) |
|---|---|---|---|---|
| 3B | neutral | 0.977 | 0.736 | — |
| 3B | other | 0.973 | 0.705 | 0.123 |
| 3B | query | 0.866 | 0.571 | 0.218 |
| 7B | neutral | 0.996 | 0.494 | — |
| 7B | other | 0.989 | 0.452 | 0.077 |
| 7B | query | 0.985 | 0.490 | 0.149 |
| Gemma | neutral | 0.897 | 0.552 | — |
| Gemma | other | 0.904 | 0.563 | 0.134 |
| Gemma | query | 0.866 | 0.521 | 0.199 |

Reference no-hint cells: absent refusal 0.904 (3B), 0.958 (7B), 0.843 (Gemma);
sufficient gold accuracy 0.762, 0.667 and 0.567. In the absent-evidence arm
the `query` sentence is followed by a stated-colour answer in 0.130 (3B),
0.008 (7B) and 0.077 (Gemma) of scenes, against 0.011 or less for `other` in
3B and Gemma.

Four things follow. First, **rotation is not specific to the query**. In 3B
the neutral sentence moves the evidence axis 42.7 degrees, as far as the `nl`
conflict axis sits from `v0` (44.2), and an assertion about another object
moves it 11.6 degrees *less* than the neutral sentence; adding the queried
category brings it back to 38.5, still short of the neutral sentence. 7B
repeats the order with smaller angles (25.0, 28.1, 30.9, 35.1), though there
the neutral sentence reaches only 25.0 of the conflict's 35.1 degrees. Gemma
is the one model where the angle rises monotonically with query relevance,
but over a 6-degree range (10.1, 14.1, 15.2, 16.3). A sentence with no object
and no colour therefore turns the axis 25--43 degrees, 62--97% of the way to
the conflict angle in every model, so the large displacement in Sections 2--4
cannot be read as query-specific reconfiguration, and the angle is not the
quantity that tracks query relevance.

Second, **the query-linked signature is compression**. The length ratio falls
monotonically with query relevance in all three models: 0.822, 0.804, 0.604,
0.589 (3B); 0.730, 0.691, 0.596, 0.568 (7B); 1.000, 0.897, 0.877, 0.885
(Gemma). The constant-colour query sentence reproduces almost all of the
conflict row's shortening: changing the asserted colour between arms adds only
0.015 (3B), 0.028 (7B) and nothing at all in Gemma, where the conflict axis is
0.008 *longer* than the constant-colour query axis. The paired query-minus-other
contrast is significant in 3B and 7B in both geometry measures, and not in
Gemma in length (-0.02 [-0.05,+0.01]) and only marginally in angle (+1.0 deg). Compression is therefore produced by an assertion about the
queried category, not by the colour conflict; the conflict manipulation adds a
small increment.

Third, **the direction agrees**. The constant-colour query axis is nearly
collinear with the conflict axis (cosine 0.976, 0.988, 0.997), while the
neutral and other-object axes sit further away (0.795 / 0.865, 0.874 / 0.921,
0.964 / 0.978). Whatever the conflict manipulation does to the axis, the query
sentence alone does nearly the same, without ever changing its colour.

Fourth, **behaviour separates them, and the query sentence is not inert**.
Under sufficient evidence gold accuracy falls in 3B from 0.762 (no hint) to
0.736 (neutral), 0.705 (other), 0.571 (query) and 0.567 (`nl` conflict), and
the model reports the other object's colour 12.3% of the time, so the
control sentences are not neutral in behaviour either. In 7B accuracy is flat
across the three controls (0.494, 0.452, 0.490) while adoption rises from
7.7% (`other`) to 14.9% (`query`) and refusal under evidence is high
throughout (0.35--0.46): for this model the readout policy, not the axis,
sets accuracy. In Gemma the first two controls are flat (0.552, 0.563) and
only the query sentence lowers accuracy to 0.521 and raises adoption to 19.9%.
The query sentence also lowers absent-arm refusal relative to the other two
controls (3B 0.866 against 0.973--0.977, Gemma 0.866 against 0.897--0.904, 7B
0.985 against 0.989--0.996): naming the queried category makes the model
willing to answer when the evidence is absent, and those answers repeat the
asserted colour 13.0% (3B), 0.8% (7B) and 7.7% (Gemma) of the time.

Caveats. The neutral and other rows are evidence axes measured under a
sentence that is identical in both arms; the query row holds the colour
constant but names the arm's queried category; the conflict row differs from
the query row only in the sufficient arm's colour. The four rows are compared
as displacements of the same reference direction, not a variance
decomposition, and the conflict row's intervals are absent because it reuses
the fixed main-battery axis. The `query`
sentence is always wrong for a present scene, so its behavioural cost bundles
query mention with a factual error, and holding the colour constant removes
colour *change*, not colour *presence*. The `other` sentence names a colour
taken from the scene's absent annotation, which can coincide with the queried
object's gold; the 0.123 rate is therefore an upper bound on colour leakage,
not a pure error rate. Gemma separates query from other weakly in length, so
the object-specific reading is strongest for the Qwen backbones.

Implication for the reviewer's question: the conflict effect is compression of
the evidence axis, and the rotation is a correlate of rewriting the prompt
that the conflict sentence shares with an assertion about any object --
including the queried one. This sharpens the reported law: the length ratio,
not the angle, is the quantity that tracks conflict accuracy across forms
(Section 7.5) and across query relevance.

## 9. What this establishes

1. The phenomenon is not an artifact of the bracket. Five forms without a
   bracketed prefix, including a two-turn dialogue, reproduce the dilution,
   the rotation, the hint-pair alignment and the frozen-probe transfer in
   Qwen 3B and 7B, and the same five forms reproduce it in Gemma-3-4B. The
   bracketed form is not what generates the geometry; behaviourally it is
   one form among six, and its rank varies by family (middle of the Qwen 3B
   range, the strongest in Gemma).
2. The compression is tied to an assertion about the queried category;
   rotation is not. Section 8 adds three controls in all three models. In 3B
   a neutral sentence moves the evidence axis 42.7 degrees, as far as the
   `nl` conflict axis sits from `v0` (44.2), so rotation is a correlate of
   rewriting the prompt. Compression, by contrast, is graded by query
   relevance: the axis shortens to 0.822 (neutral), 0.804 (other object),
   0.604 (queried category, constant colour) and 0.589 (conflict) of the
   no-hint length in 3B, with the same order in 7B
   (0.730/0.691/0.596/0.568) and Gemma (1.000/0.897/0.877/0.885). The
   constant-colour sentence reproduces nearly all of the conflict row's
   shortening and its axis is nearly collinear with the conflict axis (cosine
   0.976/0.988/0.997), so the colour change contributes only a small further
   increment. What changes with position is whether the assertion reaches the
   output. 3B adoption varies
   20-fold across forms while the axis geometry stays in a narrow band; 7B
   adoption is near zero in every form while the geometry still moves; Gemma
   reverses the behavioural ordering while keeping the geometry. The placement
   effect lives in the
   readout policy, not in the axis.
3. The representation is there even when behaviour is not: in Qwen the stated
   colour is decodable across cells in 11/12 model-by-form cells, including
   forms the model ignores. In Gemma a shared cross-cell colour code is
   demonstrable only in the two forms the model acts on (std 0.85, claim
   0.66); within-cell decoding stays high (0.71--0.99) but can lean on
   category-level colour priors, so in the other four forms the axis moves
   without a demonstrable shared colour code.
4. The form-level quantitative link is to conflict accuracy, not adoption.
   Across the six forms the evidence-axis length ratio predicts wrong-hint
   accuracy at r = +0.89 in both 3B and 7B (Spearman +0.83 / +0.93) and
   r = +0.98 in Gemma (Spearman +0.94), while its relation to adoption has no
   consistent sign (3B +0.25, 7B -0.79 where adoption never exceeds 0.05,
   Gemma -0.83). The less a form dilutes the axis, the more accurately the
   model answers under conflict, in every model tested.
5. What this does not establish: a mechanism. Attention shares, the
   colour-swap direction and the agreeing-hint control bound the
   possibilities (the rotation is not a linear colour offset, and the
   conflict cost is not symmetric with the agreement gain) but do not
   identify the weights that apply the transformation.

## Files

- `runs/prompt_forms_3b.jsonl/.npz`, `runs/prompt_forms_7b.jsonl/.npz`,
  `runs/prompt_forms_gemma.jsonl/.npz`: six-form battery.
- `runs/pf_correct_3b/7b.jsonl/.npz`: agreeing-hint cell (merged into the
  analysis); in Gemma the agreeing-hint cell is part of the main battery.
- `runs/pf_gemma_dialogue_fix.jsonl/.npz`: dialogue no-hint rerun with the
  neutral first turn, merged into `prompt_forms_gemma_fixed.*`.
- `runs/pf_query_control_{3b,7b,gemma}.jsonl/.npz`: neutral-sentence and
  other-object controls (Section 8); the 7B table is mirrored here as
  `pf_query_control_7b.txt`, the 3B and Gemma tables are folded into
  `results/summary.md` Section 10c.
- `runs/pf_query_constant_{3b,7b,gemma}.jsonl/.npz`: queried-category,
  constant-colour control (Section 8); outputs in
  `pf_query_constant_{3b,7b,gemma}.txt`. The text tables are mirrored into
  this directory; the raw runs stay on the persistent cloud disk, with the
  checksums in `pf_query_constant.md5`.
- `runs/attn_forms_3b.jsonl`: attention shares.
- `prompt_forms_gemma_fixed_analysis.json`,
  `prompt_forms_gemma_fixed_correct_analysis.json`,
  `prompt_forms_gemma_rotation.json`, `pf_gemma_correct.txt`,
  `form_rotation_gemma.txt`, `hint_transfer_gemma.txt`, `pf_ci_gemma.txt`.
- `prompt_forms_analyze.py`: behaviour and geometry; `pf_ci.py`: scene
  bootstrap CIs and paired contrasts; `pf_correct_analyze.py`: agreeing-hint
  cell; `form_rotation.py`: split-half reliability and rotation source;
  `hint_transfer.py`: cross-cell colour decoding; `prompt_forms_gen.py`:
  Gemma battery; `pf_query_control.py`, `pf_query_control_analyze.py`,
  `pf_query_constant.py` and `pf_query_constant_analyze.py`: Section 8
  controls.
