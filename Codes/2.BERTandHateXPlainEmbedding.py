

import transformers
from transformers import AutoTokenizer
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel
import torch


import os
# repro fix: read project root from env (was 'DataSetLocaltion/')
FOLDER_NAME = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data") + '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle
with open(FOLDER_NAME+'all__video_vosk_audioMap.p','rb') as fp:
    transCript = pickle.load(fp)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import *

model = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")



class Text_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", output_hidden_states = True)
        
    def forward(self,x,mask):
        embeddings = self.model(x, mask)
        return embeddings


tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
def tokenize(sentences, padding = True, max_len = 512):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}




import numpy as np




model2 = Text_Model().to(device).eval()  # repro fix: GPU



from tqdm import tqdm
allEmbedding ={}
for i in tqdm(transCript):
  try:
    apr = tokenize([transCript[i]])
    with torch.no_grad():
        allEmbedding[i]= (model2(apr['input_ids'].to(device), apr['attention_masks'].to(device))[2][0]).detach().cpu().numpy()
    del(apr)
  except Exception as e:
    print(f"HXP-FAIL {i}: {type(e).__name__}: {e}")  # repro fix: log instead of silent pass




len(allEmbedding)
with open(FOLDER_NAME+'all_HateXPlainembedding.p', 'wb') as fp:
    pickle.dump(allEmbedding,fp)



from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()  # repro fix: GPU



from tqdm import tqdm
allEmbedding ={}
for i in tqdm(transCript):
  try:
    inputs = tokenizer(transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding[i]= last_hidden_states[0][0].detach().cpu().numpy()
    del(outputs)
  except Exception as e:
    print(f"BERT-FAIL {i}: {type(e).__name__}: {e}")  # repro fix: log instead of silent pass


len(allEmbedding)
with open(FOLDER_NAME+'all_rawBERTembedding.p', 'wb') as fp:
    pickle.dump(allEmbedding,fp)
