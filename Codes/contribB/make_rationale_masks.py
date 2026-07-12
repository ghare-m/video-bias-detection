"""Build 100-frame ground-truth hateful masks from the annotated time-spans."""

import os
import csv
import ast
import pickle
import numpy as np

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data")
CSV = os.path.join(ROOT, "HateMM_annotation.csv")
IMG = os.path.join(ROOT, "Dataset_Images")
OUT = os.path.join(ROOT, "rationale_masks.p")
NFRAMES = 100


def hms(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def main():
    masks = {}
    n_skip = 0
    covs = []
    with open(CSV, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["label"].strip().lower() != "hate":
                continue
            stem = os.path.splitext(r["video_file_name"].strip())[0]
            d = os.path.join(IMG, stem)
            if not os.path.isdir(d):
                n_skip += 1
                continue
            n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
            if n == 0:
                n_skip += 1
                continue
            try:
                spans = [(hms(a), hms(b)) for a, b in ast.literal_eval(r["hate_snippet"])]
            except Exception:
                n_skip += 1
                continue
            step = max(1, n // NFRAMES)
            mask = np.zeros(NFRAMES, dtype=bool)
            for j in range(NFRAMES):
                sec = j * step
                if sec >= n:            # past the end of the (padded) video
                    break
                if any(a <= sec < b for a, b in spans):
                    mask[j] = True
            cov = float(mask.mean())
            covs.append(cov)
            masks[stem] = {"mask": mask, "coverage": cov, "n_frames": n, "step": step}

    with open(OUT, "wb") as fp:
        pickle.dump(masks, fp)
    covs = np.array(covs)
    print(f"hate videos with masks: {len(masks)}  (skipped {n_skip})")
    print(f"frame-mask coverage: mean {covs.mean():.2f}, median {np.median(covs):.2f}")
    print(f"low-coverage (<0.5) videos: {(covs < 0.5).sum()}  |  high (>=0.5): {(covs >= 0.5).sum()}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
