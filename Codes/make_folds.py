"""make_folds.py  (NEW — not part of the original HateMM repo)

The original code loads two pickle files that the dataset/repo never ships:
  - final_allNewData.p : a single train/val/test split
  - allFoldDetails.p   : 5-fold stratified CV splits (what scripts 6/7/8/9 iterate over)

This script regenerates both deterministically from HateMM_annotation.csv so the
reproduction is fair and repeatable (seed=2021, matching the paper's fix_the_random).

Input  : $HATEMM_ROOT/HateMM_annotation.csv   (cols: video_file_name,label,hate_snippet,target)
Output : $HATEMM_ROOT/final_allNewData.p
         $HATEMM_ROOT/allFoldDetails.p
         $HATEMM_ROOT/fold_ids.json           (human-readable record of every split)

Keys are video STEMS without extension (e.g. 'hate_video_1'), matching how every
downstream script keys frames/features. Labels: Hate -> 1, Non Hate -> 0.

Pickle structure (exactly what scripts 3/4/5/6/7/8/9 expect):
  final_allNewData.p -> {'train': (ids, labels), 'val': (ids, labels), 'test': (ids, labels)}
  allFoldDetails.p   -> {'fold1': {'train':(ids,labels),'val':(...),'test':(...)}, ... 'fold5': ...}
"""

import os
import csv
import json
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 2021
ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data")
CSV_PATH = os.path.join(ROOT, "HateMM_annotation.csv")

LABEL_MAP = {"hate": 1, "non hate": 0}


def load_annotation():
    """Return (ids, labels) as parallel lists; ids are stems, labels are 0/1 ints."""
    ids, labels = [], []
    with open(CSV_PATH, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            name = row["video_file_name"].strip()
            stem = os.path.splitext(name)[0]            # 'hate_video_1.mp4' -> 'hate_video_1'
            lab = LABEL_MAP[row["label"].strip().lower()]
            ids.append(stem)
            labels.append(lab)
    return ids, labels


def subset(ids, labels, idx):
    """Index into (ids, labels) by an array of positions -> (list, list)."""
    return [ids[i] for i in idx], [labels[i] for i in idx]


def main():
    ids, labels = load_annotation()
    ids = np.array(ids)
    labels = np.array(labels)
    print(f"Loaded {len(ids)} videos | hate={int(labels.sum())} non-hate={int((labels==0).sum())}")

    # ---- 1) single 70/10/20 split (final_allNewData.p) ----
    idx = np.arange(len(ids))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.20, stratify=labels, random_state=SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.125, stratify=labels[train_idx], random_state=SEED
    )  # 0.125 of the remaining 80% ≈ 10% overall -> 70/10/20
    final = {
        "train": subset(ids, labels, train_idx),
        "val": subset(ids, labels, val_idx),
        "test": subset(ids, labels, test_idx),
    }
    with open(os.path.join(ROOT, "final_allNewData.p"), "wb") as fp:
        pickle.dump(final, fp)
    print(f"final_allNewData.p: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ---- 2) 5-fold stratified CV (allFoldDetails.p) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = {}
    record = {"seed": SEED, "single_split": {k: list(v[0]) for k, v in final.items()}, "folds": {}}
    for i, (trainval_idx, test_idx) in enumerate(skf.split(idx, labels), start=1):
        # carve a validation slice (~10% overall = 1/8 of the 80% train+val) out of trainval
        tr_idx, va_idx = train_test_split(
            trainval_idx, test_size=0.125, stratify=labels[trainval_idx], random_state=SEED
        )
        key = f"fold{i}"
        folds[key] = {
            "train": subset(ids, labels, tr_idx),
            "val": subset(ids, labels, va_idx),
            "test": subset(ids, labels, test_idx),
        }
        record["folds"][key] = {
            "train": list(ids[tr_idx]), "val": list(ids[va_idx]), "test": list(ids[test_idx]),
        }
        print(f"{key}: train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} "
              f"(test hate frac={labels[test_idx].mean():.3f})")

    with open(os.path.join(ROOT, "allFoldDetails.p"), "wb") as fp:
        pickle.dump(folds, fp)
    with open(os.path.join(ROOT, "fold_ids.json"), "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)
    print("Wrote allFoldDetails.p and fold_ids.json")


if __name__ == "__main__":
    main()
