import pandas as pd
import os
import csv
import pandas as pd
import glob
# removed `import moviepy.editor as mp` (unused here, and the module was
# deleted in moviepy 2.x so it would crash on import)
import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import librosa.display
import matplotlib
matplotlib.use("Agg")  # headless backend (SSH, no display)
import matplotlib.pyplot as plt
import tarfile
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



# Helper function to generate mfccs
def extract_mfcc(path):
    audio, sr=librosa.load(path)
    # modern librosa requires keyword args (was positional: mfcc(audio, sr, ...))
    mfccs=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# read project root from env (was hardcoded './'); keep trailing '/' so
# `FOLDER_NAME+'file.p'` resolves correctly (matches original './' convention)
FOLDER_NAME = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + "/"
audioPath = FOLDER_NAME + "AudioFiles/"



import pickle
with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

# train, test split
train_list, train_label= allDataAnnotation['train']
val_list, val_label  =  allDataAnnotation['val']
test_list, test_label  =  allDataAnnotation['test']


# In[4]:


allVidList = []
allVidLab = []

allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list)

allVidLab.extend(train_label)
allVidLab.extend(val_label)
allVidLab.extend(test_label)




allAudioFeatures = {}
failedList = []




for i in tqdm(allVidList):
    try:
        aud = extract_mfcc(audioPath+i+".wav")
        allAudioFeatures[i]=aud
    except:
        print("Error", i)
        failedList.append(i)





for i in failedList:
    allAudioFeatures[i] = np.zeros(40)


import pickle
with open(FOLDER_NAME+'MFCCFeatures.p', 'wb') as fp:
    pickle.dump(allAudioFeatures,fp)


#--------------------Audio Spectrum Generation For VGG19---------------------

import os
from os import walk, listdir
import cv2
import shutil
import pandas as pd



from scipy.io.wavfile import read
from tqdm import tqdm
import matplotlib.pyplot as plt





from tqdm import tqdm
# original used an undefined `path`; spectrograms read from audioPath.
# Also ensure the output dir exists (script 4 reads these PNGs).
os.makedirs(FOLDER_NAME + "/Audio_plots/", exist_ok=True)
for selected_folder in tqdm(allVidList):
    try:
        path1 = os.path.join(audioPath, selected_folder+'.wav')

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(111)
        ax.plot(read(path1)[1])

        fig.savefig(FOLDER_NAME + "/Audio_plots/" + selected_folder + '.png')
        fig.clf()
        plt.close(fig)

    except Exception as e:
        print(e)
        continue
