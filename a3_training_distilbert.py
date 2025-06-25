import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import DistilBertModel, AdamW, DistilBertTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from a1_model_5 import HybridHateSpeechModel

def train_model():
    input_ids=torch.load("preprocessed_data_distilbert/input_ids.pt") #tokenized text input
    attention_mask=torch.load("preprocessed_data_distilbert/attention_mask.pt") #padding tokens
    target_labels=torch.load("preprocessed_data_distilbert/target_labels.pt") #binary labels for hate speech
    victim_groups=torch.load("preprocessed_data_distilbert/victim_groups.pt") #multi-label encoding for victim categories
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids_tensor=torch.tensor(input_ids,dtype=torch.long).to(device) #converts numpy arrays to pytorch tensors and move them to CPU
    attention_mask_tensor=torch.tensor(attention_mask,dtype=torch.long).to(device)
    target_labels_tensor=torch.tensor(target_labels,dtype=torch.long).to(device)
    victim_groups_tensor=torch.tensor(victim_groups,dtype=torch.float).to(device)
    dataset=TensorDataset(input_ids_tensor,attention_mask_tensor,target_labels_tensor,victim_groups_tensor)#combines tensors into single dataset
    train_mask=torch.load("preprocessed_data_distilbert/train_mask.pt").to(device)
    val_mask=torch.load("preprocessed_data_distilbert/val_mask.pt").to(device)
    test_mask=torch.load("preprocessed_data_distilbert/test_mask.pt").to(device)
    batch_size=128  
    train_data=DataLoader(dataset, batch_size=batch_size,sampler=torch.utils.data.SubsetRandomSampler(train_mask.nonzero().squeeze()),num_workers=0,pin_memory=True) #faster training
    val_data=DataLoader(dataset,batch_size=32,sampler=torch.utils.data.SubsetRandomSampler(val_mask.nonzero().squeeze()),num_workers=0,pin_memory=True) 
    test_data=DataLoader(dataset,batch_size=16,sampler=torch.utils.data.SubsetRandomSampler(test_mask.nonzero().squeeze()),num_workers=0,pin_memory=True) #smaller for detailed evaluation
    model=HybridHateSpeechModel(tfidf_vocab_size=0,num_classes=2,num_victim_groups=4).to(device)
    optimizer=AdamW(model.parameters(),lr=1e-5)
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,0.6]).to(device)) #cross entropy loss for binary classification 
    victim_loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7,1.0,1.2,1.0]).to(device)) #multi-label victime classification
    def create_dummy_edge_index(batch_size,device): #creates fully connected graph for GCN model
        row=torch.arange(batch_size,device=device).repeat(batch_size,1).flatten()
        col=torch.arange(batch_size,device=device).repeat(batch_size,1).T.flatten()
        edge_index=torch.stack([row,col],dim=0)
        return edge_index
    epochs=4
    print("Training starts now!")
    for epoch in range(epochs):
        model.train()
        total_loss=0
        print(f"\n Epoch {epoch+1}")
        for i, batch in enumerate(train_data): #iterates through batches in training data
            print(f"Processing batch {i+1}/{len(train_data)}",end="\r")
            input_ids_batch,attention_mask_batch,labels_batch,victim_group_batch=[x.to(device) for x in batch]
            optimizer.zero_grad()
            edge_index_batch=create_dummy_edge_index(input_ids_batch.size(0),device) #resets gradients before each batch
            toxicity_logits,victim_logits=model(input_ids_batch,attention_mask_batch,edge_index_batch) #forward pass
            loss_toxicity=loss_fn(toxicity_logits.squeeze(),labels_batch)
            loss_victim=victim_loss_fn(victim_logits,victim_group_batch) #higher importance to victim classification
            total_loss=(1.5*loss_toxicity)+(2.0*loss_victim)  
            total_loss.backward() #backpropagation
            optimizer.step()
        print(f"Finished Epoch {epoch+1}, Train Loss: {total_loss.item():.4f}")
        model.eval()
        val_loss=0
        val_preds,val_labels=[],[]
        val_victim_preds,val_victim_labels=[],[]
        with torch.no_grad(): #validation loop, no gradients are computed
            for batch in val_data:
                input_ids_batch,attention_mask_batch,labels_batch,victim_group_batch=[x.to(device) for x in batch]
                edge_index_batch=create_dummy_edge_index(input_ids_batch.size(0),device)
                toxicity_logits,victim_logits=model(input_ids_batch,attention_mask_batch,edge_index_batch)
                loss_toxicity=loss_fn(toxicity_logits.squeeze(),labels_batch) #validation loss
                loss_victim=victim_loss_fn(victim_logits,victim_group_batch)
                total_loss=(1.5*loss_toxicity)+(2.0*loss_victim)
                val_loss+=total_loss.item()
                toxicity_probs=torch.softmax(toxicity_logits, dim=1)
                toxicity_pred=(toxicity_probs[:,1]>0.55).long().cpu().numpy() #reduce false positives with high probability
                victim_pred=(torch.sigmoid(victim_logits)>0.4).cpu().numpy()
                val_preds.append(toxicity_pred)
                val_labels.append(labels_batch.cpu().numpy())
                val_victim_preds.append(victim_pred)
                val_victim_labels.append(victim_group_batch.cpu().numpy())
        avg_val_loss=val_loss/len(val_data)
        val_preds,val_labels=np.concatenate(val_preds),np.concatenate(val_labels)
        val_victim_preds,val_victim_labels=np.concatenate(val_victim_preds),np.concatenate(val_victim_labels)
        val_accuracy=accuracy_score(val_labels,val_preds)
        victim_accuracy=np.mean(val_victim_preds==val_victim_labels)
        print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Victim Group Accuracy: {victim_accuracy:.4f}")
    torch.save(model.state_dict(),"model_5_victim.pth")
    print("Training complete!")
if __name__ == '__main__':
    train_model()
