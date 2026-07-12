"""Contribution A: fusion model with a second head that predicts the target community."""

import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                             recall_score, precision_score, confusion_matrix)

# ---------------- config ----------------
ROOT   = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
LAMBDA = float(os.environ.get("HATEMM_LAMBDA", "1"))
EPOCHS = int(os.environ.get("HATEMM_EPOCHS", "20"))
FOLDS  = os.environ.get("HATEMM_FOLDS", "fold1,fold2,fold3,fold4,fold5").split(",")
TAG    = os.environ.get("HATEMM_TAG", "multi")
# M4 features
TEXT_PICKLE, AUDIO_PICKLE, AUDIO_DIM = "all_HateXPlainembedding.p", "vgg19_audFeatureMap.p", 1000
TEXT_DIM = 768
BATCH, LR = 10, 1e-4

def fix_the_random(seed_val=2021):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val); np.random.seed(seed_val)
    torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
fix_the_random(2021)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- model (branches identical to script 9) ----------------
class MLP(nn.Module):
    def __init__(self, in_size, h=128, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_size, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))
    def forward(self, x): return self.net(x)

class LSTMBranch(nn.Module):
    def __init__(self, input_emb_size=768, no_of_frames=100):
        super().__init__()
        self.lstm = nn.LSTM(input_emb_size, 128)
        self.fc = nn.Linear(128 * no_of_frames, 64)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

class Combined_MT(nn.Module):
    """Two heads off the shared 192-dim concat."""
    def __init__(self, n_hate=2, n_target=3):
        super().__init__()
        self.text_model  = MLP(TEXT_DIM)
        self.video_model = LSTMBranch()
        self.audio_model = MLP(AUDIO_DIM)
        self.head_hate   = nn.Linear(3 * 64, n_hate)
        self.head_target = nn.Linear(3 * 64, n_target)
    def forward(self, x_text, x_vid, x_aud):
        h = torch.cat((self.text_model(x_text), self.video_model(x_vid),
                       self.audio_model(x_aud)), dim=1)
        return self.head_hate(h), self.head_target(h)

# ---------------- data ----------------
with open(ROOT + TEXT_PICKLE, "rb") as f:  textData = pickle.load(f)
with open(ROOT + AUDIO_PICKLE, "rb") as f: audData  = pickle.load(f)
with open(ROOT + "target_labels.p", "rb") as f:
    targetData = pickle.load(f); TARGET_NAMES = targetData.pop("__names__")
with open(ROOT + "final_allNewData.p", "rb") as f: ann = pickle.load(f)
allVid = ann["train"][0] + ann["val"][0] + ann["test"][0]
vidData = {}
for i in allVid:
    with open(ROOT + "VITF/" + i + "_vit.p", "rb") as f:
        vidData[i] = np.array(pickle.load(f))

# inverse-frequency class weights for the target head (from the 3-class hate distribution)
tcounts = np.array([sum(1 for v in targetData.values() if v == c) for c in (0, 1, 2)], dtype=float)
TARGET_W = torch.FloatTensor(tcounts.sum() / (3.0 * tcounts)).to(device)
print(f"[cfg] lambda={LAMBDA} epochs={EPOCHS} folds={FOLDS} device={device} "
      f"target_counts={tcounts.tolist()} target_w={[round(x,2) for x in TARGET_W.tolist()]}")

class DS(data.Dataset):
    def __init__(self, folders, hate_labels):
        self.folders, self.hate = folders, hate_labels
    def __len__(self): return len(self.folders)
    def __getitem__(self, idx):
        f = self.folders[idx]
        try:
            xt = torch.tensor(textData[f]); xv = torch.tensor(vidData[f]); xa = torch.tensor(audData[f])
            yh = torch.LongTensor([self.hate[idx]])
            yt = torch.LongTensor([targetData[f]])   # -1 for non-hate (masked in loss/eval)
        except Exception:
            return None
        return xt, xv, xa, yh, yt

def collate(b):
    b = [x for x in b if x is not None]
    return torch.utils.data.dataloader.default_collate(b)

# ---------------- metrics ----------------
def hate_metrics(yt, yp):
    try:
        return {"accuracy": accuracy_score(yt, yp), "mF1Score": f1_score(yt, yp, average="macro"),
                "f1Score": f1_score(yt, yp, labels=np.unique(yp)),
                "auc": auc(*roc_curve(yt, yp)[:2]),
                "precision": precision_score(yt, yp, labels=np.unique(yp)),
                "recall": recall_score(yt, yp, labels=np.unique(yp))}
    except Exception:
        return {"accuracy": 0, "mF1Score": 0, "f1Score": 0, "auc": 0, "precision": 0, "recall": 0}

def target_metrics(yt, yp):
    if len(yt) == 0:
        return {"accuracy": 0, "mF1Score": 0, "perclass_f1": [0, 0, 0]}
    return {"accuracy": accuracy_score(yt, yp),
            "mF1Score": f1_score(yt, yp, labels=[0, 1, 2], average="macro"),
            "perclass_f1": f1_score(yt, yp, labels=[0, 1, 2], average=None).tolist(),
            "perclass_prec": precision_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "perclass_rec": recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0).tolist()}

def evaluate(model, loader):
    model.eval()
    yh_t, yh_p, yt_t, yt_p = [], [], [], []
    with torch.no_grad():
        for xt, xv, xa, yh, yt in loader:
            xt, xv, xa = xt.float().to(device), xv.float().to(device), xa.float().to(device)
            oh, ot = model(xt, xv, xa)
            ph = oh.argmax(1).cpu().numpy(); pt = ot.argmax(1).cpu().numpy()
            yh = yh.view(-1).numpy(); yt = yt.view(-1).numpy()
            yh_t += yh.tolist(); yh_p += ph.tolist()
            mask = yt >= 0            # target metrics on hate videos only
            yt_t += yt[mask].tolist(); yt_p += pt[mask].tolist()
    return hate_metrics(yh_t, yh_p), target_metrics(yt_t, yt_p), (yt_t, yt_p, yh_t, yh_p)

# ---------------- train ----------------
with open(ROOT + "allFoldDetails.p", "rb") as f: folds = pickle.load(f)
params  = {"batch_size": BATCH, "shuffle": True,  "num_workers": 2, "pin_memory": True}
vparams = {"batch_size": BATCH, "shuffle": False, "num_workers": 2, "pin_memory": True}
HATE_W = torch.FloatTensor([0.41, 0.59]).to(device)

results = {}
for fold in FOLDS:
    fix_the_random(2021)
    tr, trl = folds[fold]["train"]; va, val = folds[fold]["val"]; te, tel = folds[fold]["test"]
    trL = data.DataLoader(DS(tr, trl), collate_fn=collate, **params)
    vaL = data.DataLoader(DS(va, val), collate_fn=collate, **vparams)
    teL = data.DataLoader(DS(te, tel), collate_fn=collate, **vparams)
    model = Combined_MT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1; best = None
    for ep in range(EPOCHS):
        model.train()
        for xt, xv, xa, yh, yt in trL:
            xt, xv, xa = xt.float().to(device), xv.float().to(device), xa.float().to(device)
            yh = yh.view(-1).to(device); yt = yt.view(-1).to(device)
            opt.zero_grad()
            oh, ot = model(xt, xv, xa)
            loss = F.cross_entropy(oh, yh, weight=HATE_W)
            m = yt >= 0
            if m.sum() > 0:
                loss = loss + LAMBDA * F.cross_entropy(ot[m], yt[m], weight=TARGET_W)
            loss.backward(); opt.step()
        te_h, te_t, te_raw = evaluate(model, teL)
        va_h, va_t, _      = evaluate(model, vaL)
        if va_h["mF1Score"] > best_val:                 # select on val HATE macro-F1
            best_val = va_h["mF1Score"]
            best = {"test_hate": te_h, "test_target": te_t, "val_hate": va_h, "val_target": va_t,
                    "target_true": te_raw[0], "target_pred": te_raw[1],
                    "hate_true": te_raw[2], "hate_pred": te_raw[3]}
    results[fold] = best
    print(f"[{fold}] hate mF1={best['test_hate']['mF1Score']:.3f}  "
          f"target mF1={best['test_target']['mF1Score']:.3f}  "
          f"target/class F1={[round(x,2) for x in best['test_target']['perclass_f1']]}")

os.makedirs(ROOT + "../runs/contribA", exist_ok=True)
out = ROOT + f"../runs/contribA/foldWiseRes_{TAG}.p"
with open(out, "wb") as f: pickle.dump({"results": results, "target_names": TARGET_NAMES,
                                        "lambda": LAMBDA}, f)
# quick fold-mean summary
import numpy as np
h = np.mean([results[f]["test_hate"]["mF1Score"] for f in FOLDS])
t = np.mean([results[f]["test_target"]["mF1Score"] for f in FOLDS])
print(f"\nMEAN over folds:  hate mF1={h:.3f}   target mF1={t:.3f}   (lambda={LAMBDA})")
print("Wrote", out)
