#!/bin/bash
# Contribution A: single-task baseline (lambda=0) then multi-task (lambda=1), on GPU.
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm
export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data
cd /home/gharem/Work/Dissertation/HateMM
mkdir -p logs runs/contribA
rm -f logs/contribA_ALLDONE.flag

echo "[$(date +%H:%M:%S)] single-task baseline (lambda=0)"
HATEMM_LAMBDA=0 HATEMM_TAG=single python Codes/contribA/train_multitask.py > logs/contribA_single.log 2>&1

echo "[$(date +%H:%M:%S)] multi-task (lambda=1)"
HATEMM_LAMBDA=1 HATEMM_TAG=multi  python Codes/contribA/train_multitask.py > logs/contribA_multi.log 2>&1

echo "[$(date +%H:%M:%S)] DONE" | tee logs/contribA_ALLDONE.flag
