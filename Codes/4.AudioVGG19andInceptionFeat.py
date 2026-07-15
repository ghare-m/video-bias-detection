"""4.AudioVGG19andInceptionFeat.py  (REPRO REWRITE)

The original script here was non-functional: undefined `device` / `num_video_features` /
`num_audio_features` / `vidFeatureMap`, a nonsensical PCA reshaping block, a wrong audio path
(`/Audio/Audio_plots/` vs the `/Audio_plots/` that script 3 writes), and an InceptionV3 video
branch that is OUT OF SCOPE for our reproduction (model V2 dropped).

This rewrite keeps only the part we need — the **AudioVGG19 (A2)** feature used by the paper:
each audio spectrogram PNG (from script 3) → VGG19 (ImageNet-pretrained) → the **1000-d** output
vector, exactly as the paper describes ("1000 dimensional feature vectors ... pre-trained VGG-19").
Image preprocessing matches the original (Resize 224, ToTensor, Normalize(0.5, 0.5)).

Input  : $HATEMM_ROOT/Audio_plots/<stem>.png   (1068 — produced by script 3)
         $HATEMM_ROOT/final_allNewData.p        (the full video list)
Output : $HATEMM_ROOT/vgg19_audFeatureMap.p     ({stem: [1000 floats]})
         Missing spectrograms (the 15 no-audio videos) are zero-filled so all 1083 are covered.
"""

import os
import pickle

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

FOLDER_NAME = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
AUDIO_PLOTS = FOLDER_NAME + "Audio_plots/"
OUT_PATH = FOLDER_NAME + "vgg19_audFeatureMap.p"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19 pretrained on ImageNet; its 1000-class output is used as the 1000-d audio feature.
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device).eval()

# Same preprocessing the original used for the audio spectrogram images.
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Full video list (train+val+test) from the split file.
with open(FOLDER_NAME + "final_allNewData.p", "rb") as fp:
    ann = pickle.load(fp)
allVidList = []
for split in ("train", "val", "test"):
    allVidList.extend(ann[split][0])
print(f"{len(allVidList)} videos to featurize")


def vgg19_feature(png_path):
    img = Image.open(png_path).convert("RGB")          # matplotlib PNGs are RGBA -> force 3-channel
    x = transform(img).unsqueeze(0).to(device)          # (1, 3, 224, 224)
    with torch.no_grad():
        out = vgg19(x)                                  # (1, 1000)
    return out.view(-1).cpu().tolist()


audFeatureMap = {}
missing = []
for stem in tqdm(allVidList):
    png = os.path.join(AUDIO_PLOTS, stem + ".png")
    if os.path.exists(png):
        try:
            audFeatureMap[stem] = vgg19_feature(png)
        except Exception as e:
            print(f"VGG19-FAIL {stem}: {type(e).__name__}: {e}")
            missing.append(stem)
    else:
        missing.append(stem)

# zero-fill videos with no spectrogram (the 15 no-audio videos) so all 1083 are covered
for stem in missing:
    audFeatureMap[stem] = [0.0] * 1000
print(f"Done. {len(audFeatureMap)} entries ({len(missing)} zero-filled).")

with open(OUT_PATH, "wb") as fp:
    pickle.dump(audFeatureMap, fp)
print(f"Wrote {OUT_PATH}")
