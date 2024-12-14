from .PatchEmbedding import PatchEmbed
from .SpatialTransformer import STN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.stn = STN()
        self.fc = nn.Linear(3 * 224 * 224, 10)
    
    def forward(self, x):
        x = self.stn(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 
        x = self.fc(x)
        return x