from transformers import DistilBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class HybridHateSpeechModel(nn.Module):
    def __init__(self, tfidf_vocab_size, num_classes, num_victim_groups):
        super(HybridHateSpeechModel, self).__init__()
        self.bert=DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_projection=nn.Linear(768,128)
        self.gcn1=GCNConv(768,256)
        self.gcn2=GCNConv(256,128)
        self.lstm=nn.LSTM(input_size=128,hidden_size=64,num_layers=1,batch_first=True,bidirectional=True)
        self.attention=nn.Linear(128,1)
        self.hate_context_refinement=nn.Linear(128,64)
        self.negative_context=nn.Parameter(torch.randn(64))
        self.fusion_weights=nn.Linear(3,3)
        self.fc_toxicity=nn.Linear(128,num_classes)
        self.fc_victim=nn.Linear(128,num_victim_groups)
    def attention_layer(self,lstm_output):
        attention_weights=torch.tanh(self.attention(lstm_output))
        attention_weights=F.softmax(attention_weights,dim=1)
        attended_output=torch.sum(attention_weights*lstm_output,dim=1)
        return attended_output
    def forward(self,input_ids,attention_mask,edge_index):
        batch_size=input_ids.shape[0]
        bert_output=self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        bert_output_reduced=self.bert_projection(bert_output[:,0,:])
        graph_out=self.gcn1(bert_output,edge_index)
        graph_out=self.gcn2(graph_out,edge_index)
        lstm_out,_=self.lstm(graph_out)
        attended_out=self.attention_layer(lstm_out)
        hate_context=torch.tanh(self.hate_context_refinement(attended_out))  
        hate_score = torch.cosine_similarity(hate_context,self.negative_context.expand_as(hate_context),dim=-1)
        feature_stack=torch.stack([bert_output_reduced,graph_out[:,0,:],attended_out],dim=-1)  
        fusion_scores=F.softmax(self.fusion_weights(torch.ones(1,3,device=input_ids.device)),dim=-1)  
        fusion_scores=fusion_scores.unsqueeze(1)  
        fused_output=torch.sum(feature_stack*fusion_scores,dim=-1)  
        fused_output=fused_output.view(batch_size,-1)
        toxicity_logits=self.fc_toxicity(fused_output+hate_score.unsqueeze(-1))  
        victim_logits=self.fc_victim(fused_output)
        return toxicity_logits, victim_logits