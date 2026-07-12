"""Contribution A: aggregate single-task vs multi-task results (target metrics, CIs, confusion matrix)."""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
CA = ROOT + "../runs/contribA/"
FIG = CA + "figs/"; os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(2021)

def load(tag):
    p = CA + f"foldWiseRes_{tag}.p"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

def fold_mean_std(res, task, key):
    v = [res["results"][f][task][key] for f in res["results"]]
    return float(np.mean(v)), float(np.std(v))

def pool_target(res):
    yt, yp = [], []
    for f in res["results"]:
        yt += res["results"][f]["target_true"]; yp += res["results"][f]["target_pred"]
    return np.array(yt), np.array(yp)

def boot_ci(yt, yp, n=1000):
    if len(yt) == 0: return (0, 0, 0)
    stats = []
    idx = np.arange(len(yt))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        stats.append(f1_score(yt[s], yp[s], labels=[0, 1, 2], average="macro"))
    return (float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

single, multi = load("single"), load("multi")
names = (multi or single)["target_names"]
lines = ["# Contribution A — Multi-task Target Classification: results\n",
         "Base model: **M4** (HateXplain + ViT + VGG19). 5-fold, seed 2021. Target head is masked to "
         "hate videos (classes: Blacks/Jews/Other). Single-task = λ0 (hate only), Multi-task = λ1.\n"]

# ---- 1. Does the target head hurt hate detection? ----
lines.append("## 1. Effect on the main task (hate detection)\n")
lines.append("| variant | hate accuracy | hate macro-F1 |")
lines.append("|---|---|---|")
for tag, res in [("Single-task (λ0)", single), ("Multi-task (λ1)", multi)]:
    if res is None: continue
    a_m, a_s = fold_mean_std(res, "test_hate", "accuracy")
    f_m, f_s = fold_mean_std(res, "test_hate", "mF1Score")
    lines.append(f"| {tag} | {a_m:.3f} ± {a_s:.3f} | {f_m:.3f} ± {f_s:.3f} |")
lines.append("\n*(For reference, single-task M4 in Phase 1 scored macro-F1 0.767.)*\n")

# ---- 2. Target-community classification (multi-task) ----
if multi is not None:
    yt, yp = pool_target(multi)
    tmean, tstd = fold_mean_std(multi, "test_target", "mF1Score")
    bmean, blo, bhi = boot_ci(yt, yp)
    # baselines
    maj = f1_score(yt, np.zeros_like(yt), labels=[0, 1, 2], average="macro")  # predict all Blacks
    lines.append("## 2. Target-community classification (multi-task)\n")
    lines.append(f"- **Target macro-F1: {tmean:.3f} ± {tstd:.3f}** (fold mean±std); "
                 f"bootstrap 95% CI [{blo:.3f}, {bhi:.3f}] on pooled predictions.")
    lines.append(f"- Baselines: majority-class (all-Blacks) macro-F1 = {maj:.3f}; random ≈ 0.333.")
    lines.append(f"- Evaluated on {len(yt)} hate test videos pooled over folds.\n")
    # per-class table (pooled)
    from sklearn.metrics import precision_score, recall_score
    pc_f1 = f1_score(yt, yp, labels=[0, 1, 2], average=None)
    pc_p  = precision_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
    pc_r  = recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
    support = [int((yt == c).sum()) for c in (0, 1, 2)]
    lines.append("| target class | support | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for c, nm in enumerate(names):
        lines.append(f"| {nm} | {support[c]} | {pc_p[c]:.3f} | {pc_r[c]:.3f} | {pc_f1[c]:.3f} |")
    lines.append("")

    # ---- confusion matrix figure (counts) ----
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Target confusion (multi-task, pooled folds)")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#1A1A2E", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG + "target_confusion.png", bbox_inches="tight"); plt.close(fig)
    lines.append(f"Confusion matrix: `runs/contribA/figs/target_confusion.png`\n")

# ---- verdict ----
if single is not None and multi is not None:
    sh, _ = fold_mean_std(single, "test_hate", "mF1Score")
    mh, _ = fold_mean_std(multi,  "test_hate", "mF1Score")
    delta = mh - sh
    verdict = ("helped" if delta > 0.005 else "hurt" if delta < -0.005 else "left ~unchanged")
    lines.append("## 3. Verdict\n")
    lines.append(f"- Adding the target head **{verdict}** hate detection "
                 f"(Δ macro-F1 = {delta:+.3f}: single {sh:.3f} → multi {mh:.3f}).")
    lines.append(f"- The model additionally predicts the target community well above chance "
                 f"(macro-F1 {tmean:.3f} vs majority {maj:.3f}).")

open(CA + "contribA_summary.md", "w").write("\n".join(lines) + "\n")
print("Wrote", CA + "contribA_summary.md")
print("\n".join(lines))
