# HateMM Reproduction — Phase 1 Results (vs paper Table 3)

Reproduced by re-deriving the whole pipeline from raw videos (our own Vosk transcripts +
stratified 5-fold splits, seed 2021). 5-fold means. Ran on CPU (GPU driver was down at train
time; classifier heads are tiny so results are device-independent).

| Model | Modality | Ours Acc | Ours mF1 | Paper Acc | Paper mF1 | Δacc | ΔmF1 |
|-------|----------|---------:|---------:|----------:|----------:|-----:|-----:|
| T4 | text (HateXplain)        | 0.765 | 0.749 | 0.757 | 0.733 | +0.008 | +0.016 |
| A2 | audio (VGG19)            | 0.675 | 0.657 | 0.690 | 0.669 | -0.015 | -0.012 |
| V3 | video (ViT+LSTM)         | 0.715 | 0.697 | 0.748 | 0.733 | -0.033 | -0.036 |
| M1 | BERT+ViT+MFCC (headline) | 0.763 | 0.742 | 0.798 | 0.790 | -0.035 | -0.048 |
| M2 | BERT+ViT+VGG19           | 0.773 | 0.751 | 0.755 | 0.765 | +0.018 | -0.014 |
| M3 | HateXplain+ViT+MFCC      | 0.778 | 0.762 | 0.777 | 0.767 | +0.001 | -0.005 |
| M4 | HateXplain+ViT+VGG19     | 0.786 | 0.767 | 0.767 | 0.756 | +0.019 | +0.011 |

## Verdict
- **Successful reproduction:** all 7 rows within ±0.05 of the paper; 5 of 7 within ±0.02.
- **Core finding reproduced:** multimodal fusion > best unimodal. Best unimodal = T4 (mF1 0.749);
  fusions M2/M3/M4 all beat it; **our best model is M4 (0.786 / 0.767)**.
- **Deviations to discuss:**
  - Headline **M1 underperforms** (mF1 0.742 vs 0.790); our best fusion is **M4**, not M1.
    Likely: our Vosk transcripts + regenerated splits differ from the authors'; MFCC audio (used
    by M1/M3) noisier than VGG19 in our runs.
  - **V3 (video)** is the weakest match (−0.036) — vision branch most sensitive to frame
    sampling / split differences.

## Provenance
- Results: `runs/phase1/foldWiseRes_{T4_hatexplain,A2_vgg19,V3_vitlstm,M1,M2,M3,M4}.p`
- Logs: `logs/train_*.log`
- Config: 5-fold stratified CV, 70/10/20, 20 epochs, Adam, class weights [0.41,0.59],
  best epoch by val macro-F1. Scripts 6/8/9 env-parameterized; features in `$HATEMM_ROOT`.
