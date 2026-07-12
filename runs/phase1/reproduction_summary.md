# HateMM Reproduction — Phase 1 Results (vs paper Table 3)

Reproduced by re-deriving the whole pipeline from raw videos (our own Vosk transcripts +
stratified 5-fold splits, seed 2021). Scores are 5-fold means, reported as percentages.

| Model | Modality | Ours Acc | Ours mF1 | Paper Acc | Paper mF1 |
|-------|----------|---------:|---------:|----------:|----------:|
| T4 | text (HateXplain)        | 76.46% | 74.95% | 75.70% | 73.30% |
| A2 | audio (VGG19)            | 67.50% | 65.67% | 69.00% | 66.90% |
| V3 | video (ViT+LSTM)         | 71.47% | 69.70% | 74.80% | 73.30% |
| M1 | BERT+ViT+MFCC (paper's best) | 76.27% | 74.17% | 79.80% | 79.00% |
| M2 | BERT+ViT+VGG19           | 77.28% | 75.06% | 75.50% | 76.50% |
| M3 | HateXplain+ViT+MFCC      | 77.84% | 76.24% | 77.70% | 76.70% |
| M4 | HateXplain+ViT+VGG19     | 78.58% | 76.72% | 76.70% | 75.60% |

## Verdict
- **Successful reproduction:** all 7 models within 5 percentage points of the paper (5/7 within 2 pp).
- **Core finding reproduced:** multimodal fusion > best unimodal. Best unimodal = T4 (74.95%);
  fusions M2/M3/M4 all beat it; **our best model is M4 (76.72%)**.
- **Deviations to discuss:**
  - Headline **M1 underperforms** (74.17% vs 79.00%); our best fusion is **M4**, not M1. Likely: our
    Vosk transcripts + regenerated splits differ from the authors'; MFCC audio (M1/M3) is noisier
    than VGG19 in our runs.
  - **V3 (video)** is the weakest match (−3.6 pp) — vision branch most sensitive to frame
    sampling / split differences.

## Provenance
- Results: `runs/phase1/foldWiseRes_{T4_hatexplain,A2_vgg19,V3_vitlstm,M1,M2,M3,M4}.p`
- Config: 5-fold stratified CV, 70/10/20, 20 epochs, Adam, class weights [0.41, 0.59],
  best epoch by validation macro-F1.
