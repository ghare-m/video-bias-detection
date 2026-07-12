# Contribution A — Multi-task Target Classification: results

Base model: **M4** (HateXplain + ViT + VGG19). 5-fold, seed 2021. Target head is masked to hate videos (classes: Blacks/Jews/Other). Single-task = λ0 (hate only), Multi-task = λ1.

## 1. Effect on the main task (hate detection)

| variant | hate accuracy | hate macro-F1 |
|---|---|---|
| Single-task (λ0) | 0.782 ± 0.023 | 0.770 ± 0.024 |
| Multi-task (λ1) | 0.783 ± 0.020 | 0.770 ± 0.021 |

*(For reference, single-task M4 in Phase 1 scored macro-F1 0.767.)*

## 2. Target-community classification (multi-task)

- **Target macro-F1: 0.378 ± 0.092** (fold mean±std); bootstrap 95% CI [0.344, 0.450] on pooled predictions.
- Baselines: majority-class (all-Blacks) macro-F1 = 0.285; random ≈ 0.333.
- Evaluated on 431 hate test videos pooled over folds.

| target class | support | precision | recall | F1 |
|---|---|---|---|---|
| Blacks | 321 | 0.780 | 0.925 | 0.846 |
| Jews | 67 | 0.414 | 0.179 | 0.250 |
| Other | 43 | 0.143 | 0.070 | 0.094 |

Confusion matrix: `runs/contribA/figs/target_confusion.png`

## 3. Verdict

- Adding the target head **left ~unchanged** hate detection (Δ macro-F1 = -0.001: single 0.770 → multi 0.770).
- The model additionally predicts the target community well above chance (macro-F1 0.378 vs majority 0.285).
