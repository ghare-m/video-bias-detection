# Code layout

Run every script from the repo root with `HATEMM_ROOT` pointing at the data directory
(see the top-level `README.md`). Scripts read/write features and results under `$HATEMM_ROOT`.

```
Codes/
  preprocessing/
    extract_frames.py     # 1 fps frames  -> Dataset_Images/<stem>/
    extract_audio.py      # 16 kHz mono wav per video -> AudioFiles/
    make_transcripts.py   # Vosk transcripts -> all__video_vosk_audioMap.p
    make_folds.py         # train/val/test + 5-fold splits (seed 2021)
  features/
    text.py               # HateXplain + BERT transcript embeddings
    audio_mfcc.py         # MFCC features + spectrogram images
    audio_vgg19.py        # VGG19 features from the spectrograms
    video_vit.py          # ViT per-frame features -> VITF/<stem>_vit.p
    models.py             # HateXplain model wrapper (used by text.py)
  reproduction/
    train_unimodal.py     # text/audio ANN baselines (T4, A2)
    train_vision_lstm.py  # ViT + LSTM vision model (V3)
    train_fusion.py       # multimodal fusion (M1-M4)
  contribA/               # target-community classification (multi-task)
    make_target_labels.py, train_multitask.py, report.py
  contribB/               # explainability vs human rationale spans
    make_rationale_masks.py, explain.py, report.py
```

Training scripts are configured through environment variables (e.g. `HATEMM_TEXT`,
`HATEMM_AUDIO`, `HATEMM_AUDIO_DIM`, `HATEMM_FOLDS`, `HATEMM_EPOCHS`, `HATEMM_TAG`,
`HATEMM_LAMBDA`); see `run_remaining_models.sh`, `run_contribA.sh`, `run_contribB.sh`.
