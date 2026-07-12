# Project Context: Video Bias Detection (Dissertation)

> **Read this in full before starting any work.** This document is the source of truth for the project goal, the baseline being reproduced, the known gotchas in the original code, and the contributions to be built on top.

> ⚠️ **LIVE STATUS lives in `progress.md` — read it right after this file.** As of 2026-07-14:
> **PHASE 1 REPRODUCTION COMPLETE & successful** — all 7 models within ±0.05 of paper Table 3
> (5/7 within ±0.02); fusion>unimodal reproduced (best = M4 0.786/0.767). Full table + caveats in
> `runs/phase1/reproduction_summary.md`. Env built (Miniforge `hatemm`, Python 3.10, torch
> 2.6+cu124, **transformers pinned 4.46.3**); all features extracted & verified. **Caveat: the VM
> GPU driver was DOWN at train time so training ran on CPU (fine; heads are tiny) — restore the
> driver (reboot/admin) before Phase 2, which needs GPU.** Next: **Phase 2/3 contributions
> (§8–9).** A few "known issues" below are resolved/superseded (e.g. **§6.5 bert-as-service is
> wrong** — script 2 already uses HuggingFace `transformers`). Env:
> `source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm` then
> `export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data`.

---

## 1. Who I am and what I'm doing

I am an MSc Computer Science student at Trinity College Dublin working on a dissertation titled **"Video Bias Detection."** Supervisor mandate: reproduce-then-improve methodology — first reproduce a published baseline with available code and dataset, then add a focused novel contribution. Target: distinction-level mark (>70%).

**Background context:** I have limited ML background but am technically comfortable. I work on a shared GPU (constrained, not unlimited). I am the human; you (Claude Code) are the coding assistant. I prefer to understand decisions before they're made, not just see code appear.

**Framing note (for write-up only, not coding):** The dissertation is framed as *Video Bias Detection*, with hate speech as the operationalisation of *bias against protected groups*. HateMM provides target-community labels, which makes that framing concrete.

---

## 2. Research questions

| # | Question | Phase |
|---|----------|-------|
| RQ1 | Can a multimodal system (text + audio + vision) reliably detect group-targeted bias in videos? | Reproduction |
| RQ2 | Can the model also predict *which protected group* is targeted? | Contribution A |
| RQ3 | Which modality, and which moments in a video, drive the model's prediction? | Contribution B |

---

## 3. The baseline being reproduced

**Paper:** Das et al., *HateMM: A Multi-Modal Dataset for Hate Video Classification*, ICWSM 2023.
- arXiv: 2305.03915
- PDF: https://ojs.aaai.org/index.php/ICWSM/article/view/22209

**Code:** `https://github.com/hate-alert/HateMM` (already cloned into this workspace)

**Dataset:** Zenodo `10.5281/zenodo.7799469`
- `hate_videos.zip` (2.3 GB) — 431 .mp4 files
- `non_hate_videos.zip` (4.1 GB) — 652 .mp4 files
- `HateMM_annotation.csv` (63 KB) — labels + target community + frame-span rationales
- License: CC-BY-4.0

**Headline numbers to reproduce (Table 3 of the paper):**

| Model | Acc | Macro-F1 |
|---|---|---|
| T4 — HateXplain (best text-only) | 0.757 | 0.733 |
| A2 — AudioVGG19 (best audio-only) | 0.690 | 0.669 |
| V3 — ViT+LSTM (best vision-only) | 0.748 | 0.733 |
| **M1 — BERT ⊙ ViT ⊙ MFCC (best fusion)** | **0.798** | **0.790** |

Class balance: 431 hate (39.8%) / 652 non-hate (60.2%). Cohen's κ inter-annotator agreement: 0.625.

Training config used by the paper:
- 5-fold stratified cross-validation; 70/10/20 train/val/test per fold
- 20 epochs, Adam, lr=1e-4, batch size 10
- Cross-entropy loss with class weights `[0.41, 0.59]`
- Model selected per-fold on best validation macro-F1
- Final metrics: mean ± std across the 5 folds

---

## 4. Repo structure and execution order

```
HateMM/
├── Codes/
│   ├── frameExtract.py                          # Step 1: extract frames @ 1fps
│   ├── 1.FastTextEmb_and_LASEREmbExtraction.py  # Step 4a: text embeddings (fastText, LASER)
│   ├── 2.BERTandHateXPlainEmbedding.py          # Step 4b: text embeddings (BERT, HateXplain)
│   ├── 3.AudioMFCC_Feat_andSpectrumGen.py       # Step 5a: MFCC audio features
│   ├── 4.AudioVGG19andInceptionFeat.py          # Step 5b: AudioVGG19 + InceptionV3 features
│   ├── 5.Model-ViT_featureExtract.py            # Step 6: ViT vision features
│   ├── 6.Vision+lstm_foldWise.py                # Vision unimodal training
│   ├── 7. 3DCNN_withFolds.py                    # 3D-CNN training
│   ├── 8. UnimodalANN_foldWise.py               # Text/audio unimodal training
│   ├── 9. MultiModalFusionModelfoldWise.py      # THE FUSION MODEL — primary target
│   ├── models.py                                # HateXplain model definition
│   └── README.md
├── Dataset/README.md
└── requirements.txt   # UNPINNED — see issues below
```

**Execution order from raw .mp4 to trained model:**

1. Download dataset from Zenodo, unzip into one folder (e.g. `AllVideos/`)
2. Run `frameExtract.py` → produces `Dataset_Images/<video_id>/frame_<n>.jpg`
3. Extract audio (.wav) from each .mp4 using moviepy or ffmpeg
4. Run Vosk ASR on each .wav → produces `all__video_vosk_audioMap.p` (`{video_id: transcript}`)
5. Run text feature scripts (1, 2) → produces `*embedding.p` pickles
6. Run audio feature scripts (3, 4) → produces `MFCCFeatures.p`, `vgg19_audFeatureMap.p`
7. Run ViT script (5) → produces per-video pickles in `VITF/<video_id>_vit.p`
8. **Generate `allFoldDetails.p` and `final_allNewData.p` ourselves** (see issues below)
9. Run fusion script (9) for the main result

---

## 5. Dataset layout — what's provided vs what we generate

**Provided by Zenodo (only):**
- Raw .mp4 files
- `HateMM_annotation.csv` with columns: video_id, label (hate/non-hate), target_community, frame-span rationales
- `readme.txt`

**Everything else we generate ourselves:**
- Per-frame JPEGs (~144K files)
- Per-video audio .wav files
- Transcripts via Vosk (~22% OOV rate per the paper — expect noisy transcripts)
- All feature pickles (text/audio/vision)
- Cross-validation fold definitions

**First thing to do after download:** open `HateMM_annotation.csv` and inspect it. Specifically I need to know:
- Exact column names
- Distribution of target_community labels (counts per class)
- Format of frame-span rationales (start/end seconds? frame indices?)
- Any video IDs in the CSV that are missing from the zips

---

## 6. Known issues in the original repo

These will trip the code on first run. Fix them as you go and document the fixes.

1. **Hardcoded paths.** Every script has `FOLDER_NAME = './'` or `'../../'`. Refactor to read from a config or environment variable. Don't run scripts from arbitrary working directories.

2. **Old librosa API.** Script 3 uses positional args: `librosa.feature.mfcc(audio, sr, n_mfcc=40)`. Modern librosa requires keyword args: `librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)`. Same for `librosa.load`.

3. **Old moviepy API.** Script 3 imports `moviepy.editor`. In moviepy 2.x this changed to `from moviepy import ...`. Pin moviepy to 1.x for least friction, OR migrate to 2.x — your call, but document it.

4. **Unpinned `requirements.txt`.** Resolve to specific versions and commit a lockfile (or `requirements.lock.txt`). Use Python 3.10 or 3.11; some deps don't yet support 3.12.

5. **`bert-as-service` dependency** in script 2 (for raw BERT embeddings) is effectively abandoned. Replace with `transformers` library directly — load `bert-base-uncased`, take the [CLS] pooler output. Cleaner and modern.

6. **HateXplain model loading.** The `models.py` defines `Model_Rational_Label` (a BERT with a rationale token head). The pretrained weights are at `Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two` on HuggingFace. Use that, don't try to retrain HateXplain from scratch.

7. **Missing pickles that scripts assume exist.** Two files are loaded by script 9 but not provided:
   - `final_allNewData.p` — expected to be a dict with keys `'train'`, `'val'`, `'test'`, each mapping to `(video_list, label_list)`. **We generate this from the CSV.**
   - `allFoldDetails.p` — expected to be a dict with keys `'fold1'..'fold5'`, each with the same train/val/test structure. **We generate this using stratified 5-fold CV.**
   Use `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)` on the labels. Within each fold, further split the training set into 70/10 train/val (so the final split is 70/10/20 per fold).

8. **The fusion script uses `torch.utils.data.dataloader.default_collate`** which only works if all samples are non-None. The `collate_fn` filters Nones — fine, but make sure the dataset's `__getitem__` returns None (not raises) on missing features, and log which video IDs are dropped.

9. **The `evalMetric` function** silently returns zeros on any exception. This will mask bugs. Replace with proper exception handling that logs the failure.

10. **Random seeds.** The fusion script sets seeds (`fix_the_random(2021)`). Audit other scripts and add seeding everywhere — fold-level reproducibility matters for the dissertation.

---

## 7. Phase 1 — Reproduction plan

**Goal:** match the paper's M1 result (acc ≈ 0.798, macro-F1 ≈ 0.790) within reasonable variance (±0.02 is fine; if we're outside ±0.05, something is wrong).

**Order of operations:**

1. **Environment setup** — Python 3.10 venv or conda env, install pinned deps, verify GPU access (`torch.cuda.is_available()`).
2. **Data acquisition** — download Zenodo files, unzip, sanity-check counts (expect 431 hate + 652 non-hate = 1083 .mp4 files).
3. **CSV inspection** — load `HateMM_annotation.csv`, print column names, label distributions, target-community distribution, sample rows. Save a notebook for this.
4. **Generate fold pickles** — write a script `make_folds.py` that produces `final_allNewData.p` and `allFoldDetails.p` deterministically (seed=2021).
5. **Small-subset smoke test** — pick 20 hate + 20 non-hate videos. Run the full pipeline (frames → audio → transcripts → all features → fusion training for 2 epochs). Goal: prove the pipeline runs end-to-end before scaling.
6. **Full feature extraction** — run all extractors on the full 1083 videos. Save logs of failures (videos that wouldn't decode, transcripts that came back empty, etc.).
7. **Train unimodal baselines** — T4 (HateXplain), V3 (ViT+LSTM), A1 (MFCC). Compare against the paper's Table 3.
8. **Train fusion M1** — BERT ⊙ ViT ⊙ MFCC, 5 folds. This is the headline result.
9. **Write up reproduction results** — mean ± std across folds for each model, side-by-side with paper numbers.

**What I want logged for every model run:**
- Per-fold accuracy, macro-F1, hate-class precision/recall/F1, AUC
- Per-epoch training and validation loss curves
- Confusion matrix
- The exact `video_id`s in train/val/test for each fold
- All hyperparameters and the git commit SHA

Save runs to a structured directory: `runs/<phase>/<model_name>/<timestamp>/{metrics.json, config.yaml, model.pt, predictions.csv, log.txt}`.

---

## 8. Phase 2 — Contribution A: Target classification (multi-task)

**Goal:** add a second prediction head that outputs the target community of a hate video, trained jointly with the hate/non-hate head.

**First decision (before coding):** open the CSV, count target labels, decide the label set. Likely move: keep the top-N most frequent targets and bucket the rest into "Other". If a class has fewer than ~20 examples we cannot reliably evaluate it. Tell me the counts and propose the bucketing — I'll confirm before you proceed.

**Architecture change (modify `9. MultiModalFusionModelfoldWise.py`):**

The existing `Combined_model.forward` concatenates the 3×64 modality vectors and passes through one `fc_output` layer to 2 classes. Change it to:

- Keep the shared 192-dim concatenated representation
- Add **two heads** off the shared representation:
  - `head_hate`: Linear(192 → 2) — binary hate/non-hate
  - `head_target`: Linear(192 → K) — K target classes (K decided above; include a "no target" class for non-hate videos, OR mask the loss when label is non-hate)

**Loss function:**
- `L = L_hate + λ * L_target`
- `L_target` is masked to zero for non-hate videos (we don't predict targets on non-hate)
- Start with `λ = 1.0`, tune on validation
- Class weights for `L_target` computed from inverse class frequency

**Evaluation:**
- Hate task: same metrics as baseline (compare to M1)
- Target task: macro-F1 across target classes, per-class precision/recall, full confusion matrix
- Bootstrap CIs (1000 resamples) because some classes will be small
- Compare joint model's hate-detection performance against the single-task baseline — does multi-task help or hurt the main task?

**Output:** a "results_target.json" per fold + a confusion matrix plot per fold + aggregated table.

---

## 9. Phase 3 — Contribution B: Explainability

**Goal:** for each prediction, answer two questions:
- **(B1) Which modality drove the prediction?** (text vs audio vs vision)
- **(B2) Which seconds/frames of the video were most influential?** (temporal localisation)

**The killer feature:** the dataset includes human-annotated frame-span rationales. We can evaluate B2's predicted important regions against these ground-truth spans using **IoU** and **Span-F1**. Most explainability papers can't quantitatively evaluate against ground truth — we can.

### B1 — Modality attribution

Two methods, in order of complexity:

1. **Ablation-based (start here).** For each hate-predicted video at test time:
   - Compute prediction with all three modalities → baseline confidence p_full
   - Zero out the text vector → p_no_text. Δ_text = p_full − p_no_text
   - Same for audio and vision
   - Normalise Δs to get a (text%, audio%, vision%) breakdown
2. **Integrated Gradients (more rigorous).** Use `captum.attr.IntegratedGradients` against the concatenated 192-dim representation, then aggregate gradient magnitudes per modality slice.

Aggregate results across the test set and break down by target community. Look for patterns ("vision matters most for symbol-based hate against Group X, text matters most for slur-based hate").

### B2 — Temporal localisation

Two approaches — pick ONE, don't do both:

**Approach 1: Attention-based (cleaner architecturally).**
- Replace the LSTM in the vision branch with a Transformer encoder layer over the 100 per-frame ViT vectors
- Extract per-frame attention weights from the [CLS] token to each frame token
- Threshold to get predicted "important frame" regions
- Convert frame indices → seconds (1 fps sampling, so frame i ≈ second i, capped by actual video length)

**Approach 2: Gradient-based on existing model.**
- Keep the LSTM-based architecture
- Compute gradients of the hate-class logit w.r.t. each per-frame ViT input vector
- Take L2 norm per frame → per-frame importance score
- Threshold for span prediction

**Evaluation against rationale spans:**
- Parse the `HateMM_annotation.csv` rationale column to get ground-truth (start_sec, end_sec) tuples per video
- Convert predicted important frames to predicted (start_sec, end_sec) spans (with merging adjacent frames)
- Compute IoU between predicted and ground-truth spans per video
- Aggregate: mean IoU, span-F1 at IoU thresholds {0.1, 0.3, 0.5}
- Compare across modality-attribution methods if you've also implemented B1 attention-style attribution

**Important nuance:** the rationale spans only exist for hate videos. Restrict B2 evaluation to hate-predicted true-positive videos.

---

## 10. Environment and infrastructure

- Python 3.10 or 3.11 (NOT 3.12 yet — some deps lag)
- PyTorch with CUDA matching the system's CUDA version (verify before installing)
- Use a venv or conda env. Pin everything in a `requirements.lock.txt`
- Vosk: download the English model (`vosk-model-en-us-0.22` is the standard) and check it into a `models/` directory (gitignored)
- HuggingFace cache: set `HF_HOME` to a directory with enough disk (BERT and ViT weights aren't huge but matter on a shared filesystem)
- Run everything from the project root with explicit paths — never `cd` into subdirectories then run

**Storage budget estimate:**
- Raw videos: 6.5 GB
- Extracted frames: ~5 GB
- Extracted audio .wav: ~2 GB
- All feature pickles: ~1 GB
- Vosk model: ~2 GB
- HF model cache: ~3 GB
- **Total: budget 20 GB minimum**

**Compute budget estimate (rough, on one modern GPU):**
- Frame extraction: ~30 min (CPU-bound)
- Audio extraction: ~15 min
- Vosk transcription: 4–8 hours (slow, CPU-bound — start this early and let it run overnight)
- BERT/HateXplain embeddings: ~10 min on GPU
- ViT feature extraction: ~30–60 min on GPU
- AudioVGG19: ~30 min on GPU
- Each fusion training run (5 folds × 20 epochs): ~30–60 min on GPU

---

## 11. How I want you (Claude Code) to work with me

- **Ask before assuming.** If a script could be modified two ways and you're not sure which I'd want, ask. Don't pick silently.
- **Be precise about what's actually known vs what you're inferring.** If the paper doesn't state a hyperparameter, say so — don't make one up and present it as the paper's choice.
- **Surface problems honestly, including in this document.** If something here is wrong or out of date once you've inspected the code, tell me and propose an update.
- **Don't pad results.** If a reproduction number is below the paper, report it and analyse why. Do not silently re-run with different settings to chase the paper's number.
- **One change at a time.** When extending the model for Contribution A or B, make one architectural change, evaluate, document, then make the next. No bundled commits with multiple experiments.
- **Document everything as you go.** Each script you write or modify should have a docstring explaining what it does, what it expects as input, and what it produces. I will be writing about this code in the dissertation; comments matter.
- **Default to reproducibility.** Seed everything. Log everything. Save configs alongside results. The version of every model, the git SHA, the random seed.
- **If you find something the original paper didn't do well, flag it.** It might be a discussion point for the dissertation. Don't quietly fix it without telling me.

---

## 12. Out of scope (do not pursue without explicit go-ahead)

- Cross-dataset evaluation (e.g. testing on YouTube data)
- Retraining HateXplain from scratch
- Trying ViViT, Video-MAE, Whisper, or other heavier alternatives to the baseline encoders — interesting but not in scope for this dissertation given compute budget
- Web scraping new videos from BitChute or anywhere else
- LLM-based approaches (e.g. the LELA follow-up paper)

These are not bad ideas. They are simply not what this dissertation is committing to.

---

## 13. Quick reference — file → purpose

| File | Stage | Produces |
|------|-------|----------|
| `frameExtract.py` | Preprocessing | `Dataset_Images/<vid>/frame_*.jpg` |
| `1.FastTextEmb_and_LASEREmbExtraction.py` | Text features | `fastText_embedding.p`, `LASER_embedding.p` |
| `2.BERTandHateXPlainEmbedding.py` | Text features | `all_rawBERTembedding.p`, `all_HateXPlainembedding.p` |
| `3.AudioMFCC_Feat_andSpectrumGen.py` | Audio features | `MFCCFeatures.p` |
| `4.AudioVGG19andInceptionFeat.py` | Audio + vision features | `vgg19_audFeatureMap.p`, `inception_vidFeatureMap.p` |
| `5.Model-ViT_featureExtract.py` | Vision features | `VITF/<vid>_vit.p` |
| `6.Vision+lstm_foldWise.py` | Vision unimodal | results pickle |
| `7. 3DCNN_withFolds.py` | Vision unimodal | results pickle |
| `8. UnimodalANN_foldWise.py` | Text/audio unimodal | results pickle |
| `9. MultiModalFusionModelfoldWise.py` | **Fusion (M1)** | `foldWiseRes_*.p` |
| `models.py` | HateXplain wrapper | (imported by 2 and 9) |
| **`make_folds.py`** (we write) | Fold generation | `final_allNewData.p`, `allFoldDetails.p` |

---

*Last updated: at project start. Update this file whenever the plan changes.*
