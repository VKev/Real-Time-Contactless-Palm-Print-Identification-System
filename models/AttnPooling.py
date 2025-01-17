import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1)  

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1) 
        context = (attn_weights * x).sum(dim=1) 
        return context