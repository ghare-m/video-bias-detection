"""Contribution B: modality attribution (B1) and per-frame importance (B2) for the fusion model."""
import os, pickle, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as data

ROOT   = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
FOLDS  = os.environ.get("HATEMM_FOLDS", "fold1,fold2,fold3,fold4,fold5").split(",")
EPOCHS = int(os.environ.get("HATEMM_EPOCHS", "20"))
IG_STEPS = int(os.environ.get("HATEMM_IG_STEPS", "32"))
TOPK_PCTS = [1, 5, 10, 20, 50]
TEXT_DIM, AUD_DIM = 768, 1000

def seed(s=2021):
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model (single hate head; branches identical to reproduction) ----
class MLP(nn.Module):
    def __init__(self, d, h=128, o=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,o))
    def forward(self,x): return self.net(x)
class LSTMBranch(nn.Module):
    def __init__(self, d=768, nf=100):
        super().__init__(); self.lstm=nn.LSTM(d,128); self.fc=nn.Linear(128*nf,64)
    def forward(self,x):
        x,_=self.lstm(x); x=x.reshape(x.shape[0],-1); return self.fc(x)
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model=MLP(TEXT_DIM); self.video_model=LSTMBranch(); self.audio_model=MLP(AUD_DIM)
        self.head=nn.Linear(3*64,2)
    def forward(self,xt,xv,xa):
        return self.head(torch.cat((self.text_model(xt),self.video_model(xv),self.audio_model(xa)),1))

# ---- data ----
textData=pickle.load(open(ROOT+"all_HateXPlainembedding.p","rb"))
audData =pickle.load(open(ROOT+"vgg19_audFeatureMap.p","rb"))
tgt=pickle.load(open(ROOT+"target_labels.p","rb")); TNAMES=tgt.pop("__names__")
masks=pickle.load(open(ROOT+"rationale_masks.p","rb"))
ann=pickle.load(open(ROOT+"final_allNewData.p","rb"))
allVid=ann["train"][0]+ann["val"][0]+ann["test"][0]
vidData={i:np.array(pickle.load(open(ROOT+"VITF/"+i+"_vit.p","rb"))) for i in allVid}
folds=pickle.load(open(ROOT+"allFoldDetails.p","rb"))
HATE_W=torch.FloatTensor([0.41,0.59]).to(device)

class DS(data.Dataset):
    def __init__(self,f,l): self.f,self.l=f,l
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        f=self.f[i]
        try:
            return (torch.tensor(textData[f]),torch.tensor(vidData[f]),torch.tensor(audData[f]),
                    torch.LongTensor([self.l[i]]))
        except Exception: return None
def collate(b):
    b=[x for x in b if x is not None]; return torch.utils.data.dataloader.default_collate(b)

def feats(stem):
    xt=torch.tensor(textData[stem]).float().unsqueeze(0).to(device)
    xv=torch.tensor(vidData[stem]).float().unsqueeze(0).to(device)   # (1,100,768)
    xa=torch.tensor(audData[stem]).float().unsqueeze(0).to(device)
    return xt,xv,xa
def phate(model,xt,xv,xa):
    with torch.no_grad(): return F.softmax(model(xt,xv,xa),1)[0,1].item()

# ---- B2 attributions (per video) ----
def attr_grad(model,xt,xv,xa):
    xv=xv.clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):           # cuDNN can't backprop RNN in eval
        logit=model(xt,xv,xa)[0,1]
        g,=torch.autograd.grad(logit,xv)
    return g[0].norm(dim=1).detach().cpu().numpy()             # (100,)
def attr_ig(model,xt,xv,xa,steps=IG_STEPS):
    base=torch.zeros_like(xv)
    total=torch.zeros_like(xv)
    with torch.backends.cudnn.flags(enabled=False):           # cuDNN can't backprop RNN in eval
        for a in torch.linspace(1.0/steps,1.0,steps):
            x=(base+a*(xv-base)).detach().requires_grad_(True)
            logit=model(xt,x,xa)[0,1]
            g,=torch.autograd.grad(logit,x)
            total=total+g
    ig=((xv-base)*total/steps)[0]                              # (100,768)
    return ig.norm(dim=1).detach().cpu().numpy()               # per-frame magnitude
def attr_occ(model,xt,xv,xa,pf):
    xvB=xv.repeat(100,1,1)                                     # (100,100,768)
    idx=torch.arange(100)
    xvB[idx,idx,:]=0
    with torch.no_grad():
        p=F.softmax(model(xt.repeat(100,1),xvB,xa.repeat(100,1)),1)[:,1]
    return (pf-p).detach().cpu().numpy()                       # drop when frame removed

def faithfulness(model,xt,xv,xa,order,pf):
    """order = frame indices sorted by importance desc. Returns comp/suff drop lists over TOPK_PCTS."""
    comp,suff=[],[]
    for pct in TOPK_PCTS:
        k=max(1,int(round(pct/100*100)))
        top=order[:k]
        xr=xv.clone(); xr[0,top,:]=0                            # remove top-k
        comp.append(pf-phate(model,xt,xr,xa))                  # higher = important
        xk=torch.zeros_like(xv); xk[0,top,:]=xv[0,top,:]       # keep only top-k
        suff.append(pf-phate(model,xt,xk,xa))                  # lower = sufficient
    return comp,suff

# ---- run ----
records=[]
for fold in FOLDS:
    seed()
    tr,trl=folds[fold]["train"]; va,val=folds[fold]["val"]; te,tel=folds[fold]["test"]
    trL=data.DataLoader(DS(tr,trl),collate_fn=collate,batch_size=10,shuffle=True,num_workers=2,pin_memory=True)
    model=Fusion().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    for ep in range(EPOCHS):
        model.train()
        for xt,xv,xa,y in trL:
            xt,xv,xa=xt.float().to(device),xv.float().to(device),xa.float().to(device); y=y.view(-1).to(device)
            opt.zero_grad(); loss=F.cross_entropy(model(xt,xv,xa),y,weight=HATE_W); loss.backward(); opt.step()
    model.eval()
    n_tp=0
    for stem,lab in zip(te,tel):
        if lab!=1 or stem not in masks:      # explain hate videos with a GT mask
            continue
        xt,xv,xa=feats(stem)
        pf=phate(model,xt,xv,xa)
        pred_hate = pf>0.5
        # B1 modality ablation
        z=lambda t: torch.zeros_like(t)
        d_t=pf-phate(model,z(xt),xv,xa); d_v=pf-phate(model,xt,z(xv),xa); d_a=pf-phate(model,xt,xv,z(xa))
        d=np.clip([d_t,d_v,d_a],0,None); s=d.sum()
        modality=(d/s).tolist() if s>1e-9 else [1/3,1/3,1/3]
        # B2 attributions
        ig=attr_ig(model,xt,xv,xa); gr=attr_grad(model,xt,xv,xa); oc=attr_occ(model,xt,xv,xa,pf)
        order=np.argsort(-ig)                                  # faithfulness uses IG ranking
        comp,suff=faithfulness(model,xt,xv,xa,order,pf)
        records.append({"stem":stem,"fold":fold,"target":int(tgt[stem]),"correct":bool(pred_hate),
                        "p_full":pf,"gt_mask":masks[stem]["mask"],"coverage":masks[stem]["coverage"],
                        "modality":modality,   # B1: [text%, vision%, audio%]
                        "imp_IG":ig,"imp_grad":gr,"imp_occ":oc,
                        "faith_comp":comp,"faith_suff":suff})
        if pred_hate: n_tp+=1
    print(f"[{fold}] explained {sum(1 for r in records if r['fold']==fold)} hate videos "
          f"({n_tp} true-positive)")

os.makedirs(ROOT+"../runs/contribB",exist_ok=True)
out=ROOT+"../runs/contribB/explain.p"
pickle.dump({"records":records,"target_names":TNAMES,"topk_pcts":TOPK_PCTS},open(out,"wb"))
print(f"\nSaved {len(records)} records -> {out}")
