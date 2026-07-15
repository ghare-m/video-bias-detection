
import os
# repro fix: read project root from env (was hardcoded './'); trailing '/' so
# `FOLDER_NAME+'final_allNewData.p'` resolves correctly
FOLDER_NAME = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"




import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from tqdm import tqdm
from sklearn.metrics import *


# In[4]:


from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests


# In[ ]:


print(torch.__version__)


# In[5]:


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# repro fix: run ViT on GPU (original ran on CPU -> very slow for 100 frames x 1083 videos)
_vit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(_vit_device).eval()


# In[6]:


num_video_features = 1024
num_audio_features = 128
num_features = 256

k = 2
epochs = 1
batch_size = 1
learning_rate = 1e-4
log_interval = 1
minFrames = 100
img_x1, img_y1 = 299, 299
img_x2, img_y2 = 224, 224

begin_frame, end_frame, skip_frame = 0, minFrames, 0


# In[7]:


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#device = torch.device("cpu") 
device


# In[8]:


import pickle
with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

# train, test split
train_list, train_label= allDataAnnotation['train']
val_list, val_label  =  allDataAnnotation['val']
test_list, test_label  =  allDataAnnotation['test']


# In[9]:


allVidList = []
allVidLab = []

allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list)

allVidLab.extend(train_label)
allVidLab.extend(val_label)
allVidLab.extend(test_label)




def read_images(path, selected_folder):
    X = []
    currFrameCount = 0
    videoFrameCount = len([name for name in os.listdir(os.path.join(path, selected_folder))])
    if videoFrameCount <= minFrames:
        for i in range(videoFrameCount):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))

            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    else:
        step = int(videoFrameCount/minFrames)
        for i in range(0,videoFrameCount,step):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))
            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    return X



# set path
data_image_path = FOLDER_NAME + "/Dataset_Images/"  





import numpy as np


from tqdm import tqdm
import os

# repro fix: write VITF under the project root (was cwd-relative) and ensure it exists
vitf_dir = os.path.join(FOLDER_NAME, "VITF")
os.makedirs(vitf_dir, exist_ok=True)
for folder, label in tqdm(list(zip(allVidList, allVidLab))):
    out_p = os.path.join(vitf_dir, folder + "_vit.p")
    if os.path.exists(out_p):
        continue
    try:
        video = read_images(data_image_path, folder)
        inputs = feature_extractor(images=video, return_tensors="pt")
        # repro fix: run on GPU
        inputs = {k: v.to(_vit_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        video_features =[(last_hidden_states[i][0].detach().cpu().numpy()) for i in range(0,100)]
        with open(out_p, 'wb') as fp:
            pickle.dump(video_features,fp)
        del video
        del inputs
        del last_hidden_states
    except Exception as e:
        # repro fix: log the actual error instead of silently swallowing (context.md §6.9)
        print(f"VIT-FAIL {folder}: {type(e).__name__}: {e}")
        pass

