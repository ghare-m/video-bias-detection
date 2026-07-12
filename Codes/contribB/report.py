"""Contribution B: ERASER-style evaluation of the explanations against the human spans."""
import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
CB = ROOT + "../runs/contribB/"; FIG = CB + "figs/"; os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(2021)
BLUE, ORANGE, GREEN = "#3B75AF", "#E08214", "#2E8B57"

d = pickle.load(open(CB + "explain.p", "rb"))
R = d["records"]; TNAMES = d["target_names"]; PCTS = d["topk_pcts"]
TP = [r for r in R if r["correct"]]                       # explain the true-positive hate calls
valid = [r for r in TP if 0 < r["gt_mask"].sum() < 100]   # need both frame classes for AUPRC
print(f"records={len(R)} true-positive={len(TP)} usable(mixed mask)={len(valid)}")

def spans_from_mask(m):
    s, out = None, []
    for i, v in enumerate(list(m) + [False]):
        if v and s is None: s = i
        elif not v and s is not None: out.append((s, i)); s = None
    return out
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); uni = (a[1]-a[0])+(b[1]-b[0])-inter
    return inter/uni if uni else 0
def iou_f1(gt, pred):                                     # ERASER-style span IoU-F1 @0.5
    gs, ps = spans_from_mask(gt), spans_from_mask(pred)
    if not ps or not gs: return 0.0
    matched = sum(1 for p in ps if max((iou(p, g) for g in gs), default=0) >= 0.5)
    prec = matched/len(ps); rec = matched/len(gs)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def topk_mask(imp, k):
    m = np.zeros_like(imp, dtype=bool); m[np.argsort(-imp)[:k]] = True; return m

# ---------- B2 plausibility ----------
def plausibility(recs, method):
    auprc, ff1, iouf1 = [], [], []
    for r in recs:
        gt = r["gt_mask"]; imp = r[method]; k = int(gt.sum())
        auprc.append(average_precision_score(gt, imp))
        pred = topk_mask(imp, k)
        ff1.append(f1_score(gt, pred))
        iouf1.append(iou_f1(gt, pred))
    return np.mean(auprc), np.mean(ff1), np.mean(iouf1)
def rand_baseline(recs, n=20):
    ap, ff1 = [], []
    for r in recs:
        gt = r["gt_mask"]; k = int(gt.sum())
        ap.append(gt.mean())                              # AUPRC of random ranker = base rate
        f = [f1_score(gt, topk_mask(rng.random(100), k)) for _ in range(n)]
        ff1.append(np.mean(f))
    return np.mean(ap), np.mean(ff1)

low = [r for r in valid if r["coverage"] < 0.5]
lines = ["# Contribution B — Explainability: results\n",
         "Model: M4 fusion (single hate head). Explanations on true-positive hate videos, 5-fold, "
         "seed 2021. Evaluated vs human hateful time-spans (ERASER framework).\n"]

lines.append("## B2 — Temporal localisation (which seconds are hateful)\n")
lines.append(f"Evaluated on {len(valid)} true-positive hate videos ({len(low)} low-coverage <0.5).\n")
lines.append("**Plausibility — agreement with human spans (higher = better):**\n")
lines.append("| method | AUPRC | frame-F1 | IoU-F1@.5 | AUPRC (low-cov) |")
lines.append("|---|---|---|---|---|")
for meth, nm in [("imp_IG", "Integrated Gradients"), ("imp_grad", "Vanilla gradient"), ("imp_occ", "Occlusion")]:
    a, f, i = plausibility(valid, meth); al, _, _ = plausibility(low, meth)
    lines.append(f"| {nm} | {a:.3f} | {f:.3f} | {i:.3f} | {al:.3f} |")
rap, rf = rand_baseline(valid); rapl, _ = rand_baseline(low)
lines.append(f"| *Random baseline* | {rap:.3f} | {rf:.3f} | — | {rapl:.3f} |")
lines.append("")

# ---------- B2 faithfulness ----------
comp = np.mean([np.mean(r["faith_comp"]) for r in TP])
suff = np.mean([np.mean(r["faith_suff"]) for r in TP])
lines.append("**Faithfulness — does the model actually use those frames? (AOPC over top-{1,5,10,20,50}% by IG):**\n")
lines.append(f"- Comprehensiveness = **{comp:.3f}** (↑ better: removing the top frames drops hate confidence).")
lines.append(f"- Sufficiency = **{suff:.3f}** (↓ better: the top frames alone nearly retain the prediction).\n")

# ---------- B1 modality ----------
lines.append("## B1 — Which modality drove the prediction?\n")
def modmean(recs):
    a = np.array([r_["modality"] if "modality" in r_ else None for r_ in recs], dtype=object)
    return None
mm = np.array([r["modality"] for r in TP]) if "modality" in TP[0] else None
if mm is not None:
    o = mm.mean(0)
    lines.append(f"Overall (text / vision / audio): **{o[0]*100:.0f}% / {o[1]*100:.0f}% / {o[2]*100:.0f}%**\n")
    lines.append("| target | text % | vision % | audio % | n |")
    lines.append("|---|---|---|---|---|")
    by = {}
    for r in TP:
        by.setdefault(r["target"], []).append(r["modality"])
    for t in sorted(by):
        v = np.array(by[t]).mean(0)
        lines.append(f"| {TNAMES[t] if t in (0,1,2) else t} | {v[0]*100:.0f} | {v[1]*100:.0f} | {v[2]*100:.0f} | {len(by[t])} |")
    lines.append("")

    # figure: modality by target (grouped bar)
    cats = sorted(by); labels = [TNAMES[t] for t in cats]
    T = np.array([np.array(by[t]).mean(0) for t in cats])  # (ncat,3)
    x = np.arange(len(cats)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    for j, (nm, col) in enumerate([("Text", BLUE), ("Vision", GREEN), ("Audio", ORANGE)]):
        ax.bar(x + (j-1)*w, T[:, j]*100, w, label=nm, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("attribution %")
    ax.set_title("B1 — Modality attribution by target group"); ax.legend(frameon=False, ncol=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG + "modality_by_target.png", bbox_inches="tight"); plt.close(fig)

# ---------- figures: AUPRC bar + faithfulness curves + example timelines ----------
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
meths = [("imp_IG", "IG"), ("imp_grad", "grad"), ("imp_occ", "occ")]
vals = [plausibility(valid, m)[0] for m, _ in meths] + [rand_baseline(valid)[0]]
ax.bar(range(4), vals, color=[BLUE, GREEN, ORANGE, "#999999"])
ax.set_xticks(range(4)); ax.set_xticklabels(["IG", "grad", "occ", "random"])
ax.set_ylabel("AUPRC"); ax.set_title("B2 — localisation quality vs human spans")
for i, v in enumerate(vals): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG + "auprc_methods.png", bbox_inches="tight"); plt.close(fig)

comp_curve = np.mean([r["faith_comp"] for r in TP], 0); suff_curve = np.mean([r["faith_suff"] for r in TP], 0)
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
ax.plot(PCTS, comp_curve, "o-", color=BLUE, label="comprehensiveness ↑")
ax.plot(PCTS, suff_curve, "s-", color=ORANGE, label="sufficiency ↓")
ax.set_xlabel("top-k% frames by IG"); ax.set_ylabel("Δ hate confidence")
ax.set_title("B2 — faithfulness"); ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG + "faithfulness.png", bbox_inches="tight"); plt.close(fig)

# example timelines: a few low-coverage videos with good IG localisation
ex = sorted(low, key=lambda r: -average_precision_score(r["gt_mask"], r["imp_IG"]))[:3]
fig, axs = plt.subplots(len(ex), 1, figsize=(8, 2.1*len(ex)), dpi=200)
if len(ex) == 1: axs = [axs]
for ax, r in zip(axs, ex):
    imp = r["imp_IG"]/ (r["imp_IG"].max()+1e-9)
    ax.plot(imp, color=BLUE, label="IG importance")
    for a, b in spans_from_mask(r["gt_mask"]):
        ax.axvspan(a, b, color=ORANGE, alpha=0.25)
    ax.set_title(f"{r['stem']}  (coverage {r['coverage']:.2f}, AUPRC {average_precision_score(r['gt_mask'],r['imp_IG']):.2f}; shaded = human span)", fontsize=9)
    ax.set_yticks([]); ax.set_xlim(0, 99)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
axs[-1].set_xlabel("ViT frame index")
fig.tight_layout(); fig.savefig(FIG + "example_timelines.png", bbox_inches="tight"); plt.close(fig)

open(CB + "contribB_summary.md", "w").write("\n".join(lines) + "\n")
print("Wrote", CB + "contribB_summary.md")
print("\n".join(lines))
