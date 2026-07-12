# PROGRESS LOG — HateMM Reproduction (Video Bias Detection dissertation)

> Living status file for Claude sessions. Read `context.md` first (design/source-of-truth),
> then this file (what's actually been done). Update this after every meaningful step.
> Last updated: 2026-06-18 (end of day 1).

## ✅ CONTRIBUTION B DONE (explainability vs human time-spans)
Explains M4 fusion. New scripts: `Codes/make_rationale_masks.py`, `contribB_explain.py`,
`contribB_report.py`. Results in `runs/contribB/` (`contribB_summary.md` + figs). GPU, seed 2021,
5-fold, explained 429 hate videos (301 true-positive). Method = Integrated Gradients (primary) +
vanilla-grad + occlusion; ERASER-style eval (plausibility + faithfulness) vs human `hate_snippet` spans.
- **B2 temporal localisation** (283 usable TP videos, 94 low-coverage): on the **low-coverage subset
  (where localisation is meaningful)** IG AUPRC **51.58% vs random 21.55%** (~2.4×) — model attends to
  the right seconds. On ALL videos the gap is smaller (68.32% vs 62.98%) because GT coverage ~0.69 (little to
  localise) — the caveat we flagged. IG ≈ vanilla-grad (best); occlusion weaker. Example timelines
  (`figs/example_timelines.png`) show importance spiking inside the human span (compelling).
- **Faithfulness**: comprehensiveness 6.73%, sufficiency 7.40% (positive/modest — frames genuinely used).
- **B1 modality attribution** (ablation): overall text 48% / vision 47% / audio 5%. Vision dominates
  for **Jews** (50%, audio 12%) → plausibly symbol-based/antisemitic hate is more visual. Nice finding.
- Rationale masks: ViT-frame j ↔ second j*step; 429 hate videos (147/292 no-video excluded).
- Not yet committed/pushed to GitHub (offer pending).

## ✅ CONTRIBUTION A DONE (multi-task target classification)
Second head on M4 predicts target group (Blacks/Jews/Other), masked to hate videos, seed 2021,
5-fold. New scripts: `Codes/make_target_labels.py`, `contribA_multitask.py` (env `HATEMM_LAMBDA`
0=single/1=multi), `contribA_report.py`. Results in `runs/contribA/` (+ `contribA_summary.md`,
`figs/target_confusion.png`). Ran on GPU (driver restored).
- **Hate task unchanged**: single-task λ0 mF1 77.03% vs multi-task λ1 mF1 76.98% (Δ ≈ 0) → target
  head is "free" (≈ M4's 76.72%).
- **Target task**: macro-F1 **37.79%** (95% CI [34.40%, 45.02%]) vs majority 28.46% / random 33.33%.
  Blacks recall 92.52% (F1 84.62%); Jews F1 25.00%; Other F1 9.38% → strong on majority, weak on minority
  (severe imbalance; model over-predicts Blacks). Honest limitation → future work.
- Target label build: primary target, 3-class {Blacks 321, Jews 67, Other 43}; multi-target→primary.
- Not yet committed/pushed to GitHub (offer pending).

## ✅ PHASE 1 REPRODUCTION COMPLETE (2026-07-14)

All 7 models trained. **Successful reproduction** — all within 5 pp of paper Table 3, 5/7
within 2 pp. Core finding reproduced (fusion > best unimodal; our best = **M4 78.58%/76.72%**).
Full table + caveats in **`runs/phase1/reproduction_summary.md`**. Result pickles in
`runs/phase1/`, logs in `logs/train_*.log`.

| model | ours acc/mF1 | paper | model | ours acc/mF1 | paper |
|---|---|---|---|---|---|
| T4 | 76.46/74.95 | 75.70/73.30 | M1 | 76.27/74.17 | 79.80/79.00 |
| A2 | 67.50/65.67 | 69.00/66.90 | M2 | 77.28/75.06 | 75.50/76.50 |
| V3 | 71.47/69.70 | 74.80/73.30 | M3 | 77.84/76.24 | 77.70/76.70 |
|    |             |             | M4 | 78.58/76.72 | 76.70/75.60 |
(all values %)

Caveats to write up: headline **M1 low** (74.17% vs 79.00%) & our best fusion is M4 not M1;
**V3 weakest** match. Likely from our Vosk transcripts + regenerated splits.

**NOTE: GPU driver on the VM was DOWN at train time** (`torch.cuda.is_available()`=False,
nvidia module unloaded) — training ran on CPU (fine for these tiny heads; results identical).
Fix (reboot VM / admin restore driver) before Phase 2 contribution work, which will need GPU.

**Next: Phase 2 (Contribution A — target classification) / Phase 3 (Contribution B —
explainability). See `context.md` §8–9.**

---

## (archived) day-2 resume notes

**All feature extraction is DONE & verified. Smoke test passed. Now TRAINING the 6 models.**
T4 + A2 already done & match the paper. Remaining: **V3, M1, M2, M3, M4.**

Setup every session:
```bash
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm
export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data
export HF_HOME=$HATEMM_ROOT/hf_cache
cd /home/gharem/Work/Dissertation/HateMM
```

### Results so far vs paper (5-fold mean) — written to `runs/phase1/foldWiseRes_*.p`, logs in `logs/train_*.log`
| Model | Ours Acc / mF1 | Paper Acc / mF1 | status |
|------|----------------|------------------|--------|
| T4 HateXplain (text) | **0.765 / 0.749** | 0.757 / 0.733 | ✅ done, matches |
| A2 VGG19 (audio)     | **0.675 / 0.657** | 0.690 / 0.669 | ✅ done, matches |
| V3 ViT+LSTM (video)  | — | 0.748 / 0.733 | ⬜ TODO (script 6) |
| M1 BERT⊙ViT⊙MFCC     | — (smoke fold1/2ep was .742/.731) | **0.798 / 0.790** | ⬜ TODO (script 9) |
| M2/M3/M4 fusions     | — | .755/.777/.767 | ⬜ TODO (script 9) |

### Scripts are now ENV-CONFIGURABLE (no hand-editing between runs)
- **Script 8** (unimodal T4/A2): `HATEMM_FEAT`, `HATEMM_FEAT_DIM`, `HATEMM_EPOCHS`, `HATEMM_FOLDS`, `HATEMM_TAG`.
- **Script 9** (fusion M1–M4): `HATEMM_TEXT`, `HATEMM_AUDIO`, `HATEMM_AUDIO_DIM`, `HATEMM_EPOCHS`, `HATEMM_FOLDS`, `HATEMM_TAG`.
- Both write results to `runs/phase1/foldWiseRes_<TAG>.p` and print `metric : Mean .. STD ..` at the end.

### EXACT COMMANDS to finish (run from repo root, env set as above)
```bash
S8="Codes/8. UnimodalANN_foldWise.py"; S9="Codes/9. MultiModalFusionModelfoldWise.py"
# M1 (headline) BERT+ViT+MFCC :
HATEMM_TEXT=all_rawBERTembedding.p   HATEMM_AUDIO=MFCCFeatures.p        HATEMM_AUDIO_DIM=40   HATEMM_TAG=M1 python "$S9" 2>&1 | tee logs/train_M1.log
# M2 BERT+ViT+VGG19 :
HATEMM_TEXT=all_rawBERTembedding.p   HATEMM_AUDIO=vgg19_audFeatureMap.p HATEMM_AUDIO_DIM=1000 HATEMM_TAG=M2 python "$S9" 2>&1 | tee logs/train_M2.log
# M3 HateXplain+ViT+MFCC :
HATEMM_TEXT=all_HateXPlainembedding.p HATEMM_AUDIO=MFCCFeatures.p       HATEMM_AUDIO_DIM=40   HATEMM_TAG=M3 python "$S9" 2>&1 | tee logs/train_M3.log
# M4 HateXplain+ViT+VGG19 :
HATEMM_TEXT=all_HateXPlainembedding.p HATEMM_AUDIO=vgg19_audFeatureMap.p HATEMM_AUDIO_DIM=1000 HATEMM_TAG=M4 python "$S9" 2>&1 | tee logs/train_M4.log
```
**V3 (ViT+LSTM video) is NOT done yet and `6.Vision+lstm_foldWise.py` is NOT fixed yet** —
it still has `FOLDER_NAME='../../'` (needs ROOT + trailing '/'), reads `VITF/<stem>_vit.p` +
`allFoldDetails.p`, writes `foldWiseRes_lstmVision.p`. Fix it the same way as scripts 8/9
(env path + optional parameterize folds/epochs/tag; verify it has `fix_the_random` seeding),
then run it. Target 0.748 / 0.733.

### Notes
- All training reads precomputed features from `$HATEMM_ROOT` (text/audio/MFCC pickles + `VITF/`),
  so runs are fast (minutes). M1–M4 each load all 1083 ViT files (~30s) then train 5×20 epochs.
- Background `&` jobs do NOT survive SSH disconnect (only `screen` jobs do). The training runs
  are short; if one was interrupted, just re-run its command (idempotent).

---

## TL;DR — where we are
- **Phase 0 (Environment): ✅ DONE & verified.**
- **Phase 1 (Reproduction): IN PROGRESS.**
  - ✅ `make_folds.py` run → `final_allNewData.p` (757/109/217) + `allFoldDetails.p` (5 folds) + `fold_ids.json`.
  - ✅ `extract_audio.py` run → `data/AudioFiles/` = **1068 wavs** (16 kHz mono). 15 videos have
    no audio track (handled as zeros downstream); 2 audio-only videos (`hate_video_147`,
    `hate_video_292`, no video stream) recovered via direct ffmpeg.
  - ⏳ `make_transcripts.py` (Vosk) RUNNING in **screen `hatemm_vosk`** → `all__video_vosk_audioMap.p`
    (multi-hour — possibly overnight with the large en-us-0.22 model; resumable, checkpoints every 25).
  - ⏳ `frameExtract.py` RUNNING in **screen `hatemm_frames`** → `data/Dataset_Images/<stem>/frame_<n>.jpg`
    (was ~515/1083 at handoff; ~1 min from done). NOTE: 2 audio-only videos (147, 292) have NO
    video stream → no frames → expect 1081 dirs, not 1083. Handle in vision step.
  - ✅ Feature scripts PRE-FIXED & ready to run: **script 3** (MFCC: librosa kwargs, path→audioPath,
    Audio_plots dir) and **script 5** (ViT: ROOT env, GPU inference, VITF dir under ROOT, error logging).
  - ✅ **Text features DONE** (day 2): fixed+ran script 2 on GPU (path/syntax/GPU/error-log fixes).
    `all_HateXPlainembedding.p` + `all_rawBERTembedding.p`, both **1083 keys × 768-d, verified**.
    First patched `all__video_vosk_audioMap.p` to 1083 keys (empty string for the 15 no-audio).
    HF cache at `data/hf_cache` (set `HF_HOME` there).
  - ✅ **Audio features DONE** (day 2): script 3 (fixed: trailing-slash path, removed broken
    `moviepy.editor` import, matplotlib Agg, librosa kwargs) → `MFCCFeatures.p` (1083×40, 15 zero)
    + `Audio_plots/` (1068 PNGs). Script 4 **fully rewritten** (VGG19-audio only, dropped broken
    PCA + out-of-scope Inception) → `vgg19_audFeatureMap.p` (1083×1000, 15 zero). Both verified.
  - ✅ **Video features DONE** (day 2): script 5 (ViT, GPU, 8m49s) → `VITF/<stem>_vit.p` 100×768
    for 1081; the 2 no-video vids (147,292) zero-filled to 100×768.
  - ✅ **ALL FEATURES COMPLETE & verified — every store covers all 1083 videos:**
    HateXplain 768, rawBERT 768, MFCC 40, VGG19 1000, ViT 100×768. Completeness padding done.
  - ✅ **Smoke test passed** (M1 fold1/2ep: acc .742/mF1 .731) — full chain trains on GPU.
  - ✅ **Scripts 8 & 9 fixed + env-parameterized** (path/trailing-slash; toggles via env vars).
  - ✅ **T4 (0.765/0.749) & A2 (0.675/0.657) trained — both match paper.** (see RESUME table above)
  - ⬜ **REMAINING: V3 (fix script 6 first), M1, M2, M3, M4.** Exact commands in RESUME section above.
- Scope locked: reproduce **T4, A2, V3** unimodals + **M1–M4** fusions (M1 = BERT⊙ViT⊙MFCC is
  the headline: target Acc 0.798 / macro-F1 0.790). Skipped: T1/T2/T3, A1-as-reported, V1
  (3D-CNN), V2 (InceptionV3).
- **New scripts live directly in `Codes/`** (not a subfolder): `make_folds.py`,
  `extract_audio.py`, `make_transcripts.py`. All read `HATEMM_ROOT` env var.
- **Run scripts with:** `export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data` and
  absolute script paths (cwd is not guaranteed).
- Completeness TODO: 15 no-audio (+maybe 2 no-video) videos will be missing some features;
  before training, ensure every one of the 1083 stems appears in each feature dict (zero-fill)
  so none get silently dropped by the fusion dataloader.

---

## Environment (done)
- **No conda/ffmpeg on the VM, system Python 3.13, `sudo` locked** → installed **Miniforge**
  user-space at `~/miniforge3`.
- Env **`hatemm`**, Python **3.10.20**. Activate with:
  ```bash
  source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm
  ```
- Key pins (full list in `requirements.lock.txt`, 91 pkgs):
  - torch **2.6.0+cu124** — `torch.cuda.is_available() == True`, GPU = **NVIDIA RTX A6000 (49 GB)**
  - **transformers 4.46.3** (deliberately pinned DOWN from 5.12.1 — 5.x removed
    `ViTFeatureExtractor` and breaks the 2023 code; 4.46.3 keeps legacy imports). tokenizers
    0.20.3, huggingface-hub 0.36.2.
  - ffmpeg 8.1.2 (conda-forge), librosa 0.11.0, moviepy **2.2.1**, opencv 4.13, vosk 0.3.45,
    numpy 2.2.6 (no conflicts), pandas 2.3.3, scikit-learn 1.7.2.
  - **`fasttext` intentionally NOT installed** (T1 out of scope; avoids build pain).
- **moviepy is 2.x** → use `from moviepy import VideoFileClip` (NOT `moviepy.editor`).

## Data (done earlier)
- `data/AllVideos/` = **1083 mp4** (431 hate / 652 non-hate). CSV ↔ disk match exactly, 0 missing.
- `data/HateMM_annotation.csv` cols: `video_file_name,label,hate_snippet,target`.
- **All 431 hate videos have rationale spans** `[['HH:MM:SS','HH:MM:SS'],...]` (Contribution B gold).
- `target` is messy (plain / `['Jews']` / `Blacks,Jews`) → normalise. Usable hate targets:
  **Blacks 321, Jews 67**, rest ≤15 → Contribution-A label set = **{Blacks, Jews, Other}**.
- Raw zips already cleaned up; `data/` holds only `AllVideos/`, the CSV, `readme.txt`.

---

## NEXT STEPS (Phase 1 reproduction, in order)
1. **`make_folds.py`** → `final_allNewData.p` (70/10/20 split) + `allFoldDetails.p`
   (`StratifiedKFold(5, shuffle=True, random_state=2021)`, 70/10 train/val carved per fold).
   Key by video **stem** (no extension). Log per-fold IDs.
2. **`extract_audio.py`** → `data/AudioFiles/<stem>.wav` (moviepy 2.x / ffmpeg). Log no-audio vids.
3. **`make_transcripts.py`** (Vosk `vosk-model-en-us-0.22`) → `data/all__video_vosk_audioMap.p`.
   **START THIS FIRST — 4–8 h, CPU-bound.** Run in background.
4. **`frameExtract.py`** (set ROOT, reads `AllVideos/`) → `data/Dataset_Images/<stem>/frame_<n>.jpg`.
5. **Feature extraction (locked scope only):**
   - `2.BERTandHateXPlain*` → `all_HateXPlainembedding.p` + `all_rawBERTembedding.p`
   - `3.AudioMFCC*` → `MFCCFeatures.p` + spectrogram PNGs
   - `4.AudioVGG19*` → `vgg19_audFeatureMap.p`
   - `5.ViT*` → `VITF/<stem>_vit.p`
6. **Smoke test** on 20+20 videos, epochs=2, full chain, before scaling.
7. **Train:** `8.UnimodalANN` (T4 HateXplain, A2 VGG19), `6.Vision+lstm` (V3),
   then `9.MultiModal` ×4 (M1–M4).
8. **Reproducibility wrapper** → `runs/phase1/<model>/<ts>/{metrics.json,config.yaml,predictions.csv,log.txt}`.

`ROOT = /home/gharem/Work/Dissertation/HateMM/data` everywhere (prefer
`os.environ["HATEMM_ROOT"]` over editing `FOLDER_NAME` in each script).

---

## KNOWN CODE ISSUES to fix when running each script (verified by reading the source)
- **All scripts:** inconsistent hardcoded `FOLDER_NAME` (`'./'` / `'../../'` / `'DataSetLocaltion/'`).
- **Script 9 default = M4, NOT M1.** For M1 uncomment rawBERT + MFCC pickles AND set
  `input_size_audio` 1000→40.
- **Script 1** (skipped/scope): `fastTextEmbedding[t]`→`[i]` + stray `:` syntax error.
- **Script 2:** stray `:` syntax error; run from `Codes/` (imports `models.py`).
- **Script 3:** `librosa.feature.mfcc(audio, sr, ...)` → `mfcc(y=audio, sr=sr, n_mfcc=40)`;
  spectrogram loop uses undefined `path` → `audioPath`.
- **Script 4:** undefined `device`, `num_video_features`, `num_audio_features`, `vidFeatureMap`;
  PNG path mismatch (`Audio/Audio_plots` vs `Audio_plots`).
- **Scripts 6/7/8:** add seeding (only 9 seeds); replace silent-zero `evalMetric` except-handler.
- **No audio-extraction code exists** in the repo (script 3 only *reads* wavs) → we write
  `extract_audio.py`.

## context.md corrections (flagged, not yet applied to context.md itself)
- §6.5 (bert-as-service) is **wrong** — script 2 already uses HuggingFace `transformers`.
- §6.3 — script 3's `moviepy` import is vestigial; real task is writing audio extraction.
- §13 filename fixes: `all_fastTextEmbedding.p`/`all_LaserEmbedding.p`, `inception_vidFeatures.p`.
- Add the "script 9 defaults to M4" gotcha.

## Plan file
Full plan: `~/.claude/plans/i-am-connected-to-noble-stonebraker.md` (Phase-1 execution plan).
