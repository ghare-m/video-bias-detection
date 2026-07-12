#!/bin/bash
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate hatemm
export HATEMM_ROOT=/home/gharem/Work/Dissertation/HateMM/data
cd /home/gharem/Work/Dissertation/HateMM
mkdir -p logs runs/contribB
echo "[$(date +%H:%M:%S)] contribB full 5-fold explain (20ep, IG 32 steps)"
python Codes/contribB/explain.py > logs/contribB_explain.log 2>&1
echo "[$(date +%H:%M:%S)] DONE" | tee logs/contribB_ALLDONE.flag
