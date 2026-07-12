# Contribution B — Explainability: results

Model: M4 fusion (single hate head). Explanations on true-positive hate videos, 5-fold, seed 2021.
Evaluated vs human hateful time-spans (ERASER framework). Scores in %.

## B2 — Temporal localisation (which seconds are hateful)

Evaluated on 283 true-positive hate videos (94 low-coverage <0.5).

**Plausibility — agreement with human spans (higher = better):**

| method | AUPRC | frame-F1 | AUPRC (low-cov) |
|---|---|---|---|
| Integrated Gradients | 68.32% | 71.15% | 51.58% |
| Vanilla gradient | 68.38% | 71.03% | 52.08% |
| Occlusion | 66.59% | 67.41% | 38.64% |
| *Random baseline* | 62.98% | 62.90% | 21.55% |

On the **low-coverage subset** (where localisation is meaningful, since hateful spans otherwise
cover ~69% of a video), Integrated Gradients reaches **51.58%** vs random **21.55%** (~2.4×) — the
model attends to the right seconds. On all videos the gap is small because there is little to
localise.

**Faithfulness — does the model actually use those frames? (AOPC over top-{1,5,10,20,50}% by IG):**

- Comprehensiveness = **6.73%** (↑ better: removing the top frames drops hate confidence).
- Sufficiency = **7.40%** (↓ better: the top frames alone nearly retain the prediction).

## B1 — Which modality drove the prediction?

Overall (text / vision / audio): **47.81% / 46.95% / 5.24%**

| target | text | vision | audio | n |
|---|---|---|---|---|
| Blacks | 50.04% | 46.06% | 3.90% | 229 |
| Jews | 37.93% | 50.37% | 11.70% | 46 |
| Other | 45.65% | 48.74% | 5.61% | 26 |

Vision dominates for antisemitic ("Jews") videos → plausibly more symbol-based hate.

Figures: `runs/contribB/figs/` (auprc_methods, faithfulness, modality_by_target, example_timelines).
