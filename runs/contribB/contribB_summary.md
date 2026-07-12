# Contribution B — Explainability: results

Model: M4 fusion (single hate head). Explanations on true-positive hate videos, 5-fold, seed 2021. Evaluated vs human hateful time-spans (ERASER framework).

## B2 — Temporal localisation (which seconds are hateful)

Evaluated on 283 true-positive hate videos (94 low-coverage <0.5).

**Plausibility — agreement with human spans (higher = better):**

| method | AUPRC | frame-F1 | IoU-F1@.5 | AUPRC (low-cov) |
|---|---|---|---|---|
| Integrated Gradients | 0.683 | 0.712 | 0.218 | 0.516 |
| Vanilla gradient | 0.684 | 0.710 | 0.210 | 0.521 |
| Occlusion | 0.666 | 0.674 | 0.208 | 0.386 |
| *Random baseline* | 0.630 | 0.629 | — | 0.216 |

**Faithfulness — does the model actually use those frames? (AOPC over top-{1,5,10,20,50}% by IG):**

- Comprehensiveness = **0.067** (↑ better: removing the top frames drops hate confidence).
- Sufficiency = **0.074** (↓ better: the top frames alone nearly retain the prediction).

## B1 — Which modality drove the prediction?

Overall (text / vision / audio): **48% / 47% / 5%**

| target | text % | vision % | audio % | n |
|---|---|---|---|---|
| Blacks | 50 | 46 | 4 | 229 |
| Jews | 38 | 50 | 12 | 46 |
| Other | 46 | 49 | 6 | 26 |

