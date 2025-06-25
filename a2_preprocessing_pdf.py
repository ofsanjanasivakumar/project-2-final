import json
import numpy as np
import pandas as pd
import torch
import os
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

DATA_DIR="preprocessed_data_distilbert"
os.makedirs(DATA_DIR,exist_ok=True)
with open("dataset.json","r") as file: #hatexplain dataset
    data=json.load(file)
texts=[]
target_labels=[]
victim_groups=[]
tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased") #distilbert's tokenizer
victim_group_map = {
    "Race":["African","Arab","Asian","Caucasian","Hispanic","Indian","Indigenous","Minority"],
    "Religion":["Buddhism","Christian","Hindu","Islam","Jewish","Nonreligious"],
    "Gender":["Men","Women","Asexual","Bisexual","Heterosexual","Homosexual"],
    "Other":["Disability","Economic","Refugee","Other"]
} #maps different victim group categories to specific subgroups
def encode_victim_groups(targets):
    victim_vector=[0,0,0,0] #corresponds to [Race,Religion,Gender,Other]
    for target in targets:
        for category,subgroups in victim_group_map.items():
            if target in subgroups:
                index=["Race","Religion","Gender","Other"].index(category)
                victim_vector[index]=1
    return victim_vector
for key, value in data.items():
    text=" ".join(value["post_tokens"])
    texts.append(text)
    annotator_labels=[annotator['label'] for annotator in value['annotators']]
    if 'hatespeech' in annotator_labels or 'offensive' in annotator_labels:
        target_labels.append(1)#hate speech (1)
    else:
        target_labels.append(0)#not hate speech (0)
    victim_group_all=[]
    for annotator in value["annotators"]:
        victim_group_all.extend(annotator["target"])
    victim_group_cleaned=list(set(victim_group_all)-{"None"}) 
    if not victim_group_cleaned:
        victim_group_cleaned=["Other"] 
    victim_groups.append(encode_victim_groups(victim_group_cleaned))
bert_inputs=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors="pt")
input_ids=bert_inputs["input_ids"]
attention_mask=bert_inputs["attention_mask"] #stores tokenized data
target_labels_tensor=torch.tensor(target_labels, dtype=torch.long) #labels into pytorch tensors
victim_groups_tensor=torch.tensor(victim_groups, dtype=torch.float)
torch.save(input_ids,f"{DATA_DIR}/input_ids.pt")
torch.save(attention_mask,f"{DATA_DIR}/attention_mask.pt")
torch.save(target_labels_tensor,f"{DATA_DIR}/target_labels.pt")
torch.save(victim_groups_tensor,f"{DATA_DIR}/victim_groups.pt")
df=pd.DataFrame({"text":texts,"label":target_labels,"victim_groups":victim_groups})
df.to_csv(f"{DATA_DIR}/dataset.csv",index=False)
train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42)
train_idx,val_idx=train_test_split(train_idx,test_size=0.1,random_state=42)
train_mask=torch.zeros(len(df),dtype=torch.bool) #binary masks to indicate which samples belong to train,val,test
train_mask[train_idx]=True
val_mask=torch.zeros(len(df),dtype=torch.bool)
val_mask[val_idx]=True
test_mask=torch.zeros(len(df),dtype=torch.bool)
test_mask[test_idx]=True
torch.save(train_mask,f"{DATA_DIR}/train_mask.pt")
torch.save(val_mask,f"{DATA_DIR}/val_mask.pt")
torch.save(test_mask,f"{DATA_DIR}/test_mask.pt")
print(f"Preprocessed data has been saved in '{DATA_DIR}' successfully.")
