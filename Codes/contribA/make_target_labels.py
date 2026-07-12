"""Build 3-class target-community labels (Blacks/Jews/Other) for the hate videos."""

import os
import csv
import ast
import pickle
import collections

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data")
CSV = os.path.join(ROOT, "HateMM_annotation.csv")
OUT = os.path.join(ROOT, "target_labels.p")

NAMES = ["Blacks", "Jews", "Other"]
IDX = {"Blacks": 0, "Jews": 1, "Other": 2}


def norm_targets(t):
    """Normalise the messy target cell into a list of group strings."""
    t = (t or "").strip()
    if not t or t == "[]":
        return []
    if t.startswith("["):
        try:
            return [str(s).strip() for s in ast.literal_eval(t)]
        except Exception:
            pass
    return [s.strip() for s in t.split(",") if s.strip()]


def to3(primary):
    return primary if primary in ("Blacks", "Jews") else "Other"


def main():
    labels = {}
    dist = collections.Counter()
    n_hate = n_nonhate = n_multi = 0
    with open(CSV, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            stem = os.path.splitext(r["video_file_name"].strip())[0]
            is_hate = r["label"].strip().lower() == "hate"
            if not is_hate:
                labels[stem] = -1
                n_nonhate += 1
                continue
            n_hate += 1
            groups = norm_targets(r["target"])
            if len(groups) > 1:
                n_multi += 1
            primary = groups[0] if groups else "Other"
            cls = to3(primary)
            labels[stem] = IDX[cls]
            dist[cls] += 1

    labels["__names__"] = NAMES
    with open(OUT, "wb") as fp:
        pickle.dump(labels, fp)

    print(f"hate={n_hate}  non-hate={n_nonhate}  multi-target(hate)={n_multi}")
    print("3-class target distribution (hate videos):")
    for k in NAMES:
        print(f"   {k:8s} {dist[k]}")
    print(f"Wrote {OUT}  ({len([v for v in labels.values() if isinstance(v,int)])} labels)")


if __name__ == "__main__":
    main()
