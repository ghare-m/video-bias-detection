#!/bin/bash
# Runs the remaining reproduction models sequentially on the GPU.
# Designed to run inside a detached `screen` session so it survives SSH disconnect.
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm
export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data
export HF_HOME=$HATEMM_ROOT/hf_cache
cd /home/gharem/Work/Dissertation/HateMM
mkdir -p logs runs/phase1
rm -f logs/train_ALLDONE.flag

S6="Codes/6.Vision+lstm_foldWise.py"
S9="Codes/9. MultiModalFusionModelfoldWise.py"

echo "[$(date +%H:%M:%S)] START V3"
HATEMM_TAG=V3_vitlstm python "$S6" > logs/train_V3.log 2>&1

echo "[$(date +%H:%M:%S)] START M1 (BERT+ViT+MFCC)"
HATEMM_TEXT=all_rawBERTembedding.p    HATEMM_AUDIO=MFCCFeatures.p        HATEMM_AUDIO_DIM=40   HATEMM_TAG=M1 python "$S9" > logs/train_M1.log 2>&1

echo "[$(date +%H:%M:%S)] START M2 (BERT+ViT+VGG19)"
HATEMM_TEXT=all_rawBERTembedding.p    HATEMM_AUDIO=vgg19_audFeatureMap.p HATEMM_AUDIO_DIM=1000 HATEMM_TAG=M2 python "$S9" > logs/train_M2.log 2>&1

echo "[$(date +%H:%M:%S)] START M3 (HateXplain+ViT+MFCC)"
HATEMM_TEXT=all_HateXPlainembedding.p HATEMM_AUDIO=MFCCFeatures.p        HATEMM_AUDIO_DIM=40   HATEMM_TAG=M3 python "$S9" > logs/train_M3.log 2>&1

echo "[$(date +%H:%M:%S)] START M4 (HateXplain+ViT+VGG19)"
HATEMM_TEXT=all_HateXPlainembedding.p HATEMM_AUDIO=vgg19_audFeatureMap.p HATEMM_AUDIO_DIM=1000 HATEMM_TAG=M4 python "$S9" > logs/train_M4.log 2>&1

echo "[$(date +%H:%M:%S)] ALL DONE" | tee logs/train_ALLDONE.flag
