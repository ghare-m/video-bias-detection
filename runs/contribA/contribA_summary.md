# Contribution A — Multi-task Target Classification: results

Base model: **M4** (HateXplain + ViT + VGG19). 5-fold, seed 2021. Target head is masked to hate
videos (classes: Blacks/Jews/Other). Single-task = λ0 (hate only), Multi-task = λ1. Scores in %.

## 1. Effect on the main task (hate detection)

| variant | hate accuracy | hate macro-F1 |
|---|---|---|
| Single-task (λ0) | 78.21% ± 2.25 | 77.03% ± 2.41 |
| Multi-task (λ1) | 78.30% ± 2.04 | 76.98% ± 2.13 |

*(For reference, single-task M4 in Phase 1 scored macro-F1 76.72%.)*

## 2. Target-community classification (multi-task)

- **Target macro-F1: 37.79% ± 9.25** (fold mean±std); bootstrap 95% CI [34.40%, 45.02%].
- Baselines: majority-class (all-Blacks) macro-F1 = 28.46%; random ≈ 33.33%.
- Evaluated on 431 hate test videos pooled over folds.

| target class | support | precision | recall | F1 |
|---|---|---|---|---|
| Blacks | 321 | 77.95% | 92.52% | 84.62% |
| Jews | 67 | 41.38% | 17.91% | 25.00% |
| Other | 43 | 14.29% | 6.98% | 9.38% |

Confusion matrix: `runs/contribA/figs/target_confusion.png`

## 3. Verdict
- Adding the target head leaves hate detection **~unchanged** (single 77.03% → multi 76.98%, Δ ≈ 0).
- The model additionally predicts the target community above chance (37.79% vs majority 28.46%),
  but is strong only on the majority class (Blacks) and weak on the rare classes (Jews, Other)
  because of the severe class imbalance (321 / 67 / 43).
