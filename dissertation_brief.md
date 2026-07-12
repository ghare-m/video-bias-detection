# Dissertation Brief — Video Bias Detection (Reproducing & Extending HateMM)

Self-contained summary of the completed technical work, for the write-up and as context for
Claude Desktop. All scores are reported as **percentages** (macro-F1 unless stated). Repo:
https://github.com/ghare-m/video-bias-detection

---

## 1. Goal and research questions
Detect **group-targeted bias (hate) in videos** using text + audio + vision, then extend the
model. Method: **reproduce, then improve**.
- **RQ1 (Reproduction):** can we faithfully rebuild the multimodal HateMM classifier?
- **RQ2 (Contribution A):** can the model also predict **which protected group** is targeted?
- **RQ3 (Contribution B):** can it **explain** its decisions — which modality and which seconds —
  and do those explanations match human annotations?

Baseline paper: Das et al., *HateMM: A Multi-Modal Dataset for Hate Video Classification*, ICWSM 2023.

## 2. Data and method
- **Dataset:** HateMM — **1,083 videos** (431 hate / 652 non-hate) from BitChute. Each hate video
  also has a **target group** and human-annotated **hateful time-spans** (Cohen's κ = 0.625).
- **Pipeline:** videos → per-modality features → models.
  - Text: transcripts (Vosk ASR) → BERT and HateXplain embeddings.
  - Audio: MFCC features + **VGG-19** features from spectrogram images.
  - Vision: 1 fps frames → **ViT** per-frame features → LSTM over the clip.
  - Fusion: each modality → 64-dim → concat (192) → classifier.
- **Protocol:** 5-fold stratified CV, 70/10/20, 20 epochs, seed 2021; report mean macro-F1.
- **Environment:** conda env `hatemm` (Python 3.10, PyTorch 2.6+cu124, transformers 4.46.3).

---

## 3. RQ1 — Reproduction results (ours vs paper, macro-F1)

| Model | Modality | Ours mF1 | Paper mF1 |
|-------|----------|---------:|----------:|
| T4 | text (HateXplain)        | 74.95% | 73.30% |
| A2 | audio (VGG19)            | 65.67% | 66.90% |
| V3 | video (ViT+LSTM)         | 69.70% | 73.30% |
| M1 | BERT+ViT+MFCC (paper's best) | 74.17% | 79.00% |
| M2 | BERT+ViT+VGG19           | 75.06% | 76.50% |
| M3 | HateXplain+ViT+MFCC      | 76.24% | 76.70% |
| **M4** | **HateXplain+ViT+VGG19 (our best)** | **76.72%** | 75.60% |

**Verdict:** all 7 models within 5 percentage points of the paper (5/7 within 2 pp). The paper's
core finding — **multimodal fusion beats any single modality** — reproduced (best model **M4**).

## 4. RQ2 — Contribution A: target-community classification (multi-task)
Second head on M4 predicts Blacks / Jews / Other (masked to hate videos). Single-task (λ=0) vs
multi-task (λ=1):

- **Effect on main task:** hate macro-F1 = **77.03%** (single) vs **76.98%** (multi) → the target
  head is essentially **free** (Δ ≈ 0).
- **Target task:** macro-F1 **37.79%** (95% CI [34.40%, 45.02%]) vs majority baseline 28.46% and
  random 33.33%.
- **Per class:** Blacks F1 **84.62%** (recall 92.52%); Jews F1 25.00%; Other F1 9.38%.
- **Honest limitation:** strong on the majority class, weak on rare targets (severe imbalance:
  321 / 67 / 43) — the model over-predicts "Blacks".

## 5. RQ3 — Contribution B: explainability (evaluated vs human spans, ERASER framework)
**B1 — which modality drove it?** (leave-one-modality-out) Overall **text 47.81% / vision 46.95% /
audio 5.24%**. Vision dominates for antisemitic ("Jews") videos → plausibly more symbol-based.

**B2 — which seconds were hateful?** (Integrated Gradients over the 100 ViT frames, scored vs the
human time-spans; 283 usable true-positive videos, 94 low-coverage):

| Method | AUPRC (all) | AUPRC (low-coverage) |
|--------|------------:|---------------------:|
| Integrated Gradients | 68.32% | **51.58%** |
| Vanilla gradient | 68.38% | 52.08% |
| Occlusion | 66.59% | 38.64% |
| Random baseline | 62.98% | **21.55%** |

- **Headline:** on low-coverage videos (where localisation is meaningful), IG reaches **51.58%**
  vs random **21.55%** (~2.4×) — the model attends to the right seconds. On all videos the gap is
  small because hateful spans cover ~69% of most clips (little to localise).
- **Faithfulness:** comprehensiveness **6.73%** (removing top frames drops confidence), sufficiency
  **7.40%** (top frames alone nearly retain it) → the highlighted frames are genuinely used.

---

## 6. Honest caveats / limitations (viva material)
- **M1 (paper's headline) underperforms** for us (74.17% vs 79.00%); our best fusion is **M4**, not
  M1. Cause: our Vosk transcripts and regenerated splits differ from the authors' (they released
  neither), and MFCC audio is noisier than VGG19 in our runs.
- **Contribution A** minority-target detection is weak (class imbalance) — future work: balanced
  sampling / focal loss, or a Blacks-vs-Jews reframing.
- **Contribution B** raw metrics look high because GT spans cover ~69% of videos → the
  **low-coverage subset + random baseline** are the honest headline.

## 7. Key decisions
- Reproduced best-of-each-modality + all four fusions (not every ablation).
- transformers pinned to 4.46.3 (5.x removed `ViTFeatureExtractor`).
- Target as masked 3-class (predict only on hate videos).
- Explanation via **Integrated Gradients**, not attention ("attention is not explanation").

## 8. Repo map
```
Codes/preprocessing/  frames, audio, transcripts, folds
Codes/features/       text, audio_mfcc, audio_vgg19, video_vit, models
Codes/reproduction/   train_unimodal, train_vision_lstm, train_fusion
Codes/contribA/       make_target_labels, train_multitask, report
Codes/contribB/       make_rationale_masks, explain, report
runs/phase1/  runs/contribA/  runs/contribB/   -> result pickles, *_summary.md, figs/
```
Dataset, features, and the paper PDF are excluded from git (see `.gitignore`); download data from
Zenodo 10.5281/zenodo.7799469.

## 9. Status
Technical work (RQ1–RQ3) complete, evaluated, cleaned, and pushed. Remaining: the written thesis.
Citation fixes still to apply in the lit review: cite **VGG-19 (Simonyan & Zisserman 2015)** for
the audio features (not VGGish), and add **ERASER (DeYoung et al. 2020)** and **LSTM (Hochreiter &
Schmidhuber 1997)**.
