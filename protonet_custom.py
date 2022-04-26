## PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from einops import rearrange

class CustProtoNet(nn.Module):
    def __init__(self, in_channels, out_dim, hid_dim = 64):
        super(CustProtoNet, self).__init__()
        
        self.out_dim = out_dim
        
        self.encoder = nn.Sequential(
                                        self._conv3x3(in_channels, hid_dim),
                                        self._conv3x3(hid_dim, hid_dim),
                                        self._conv3x3(hid_dim, hid_dim),
                                        self._conv3x3(hid_dim, out_dim)
                                    )
            
    def _conv3x3(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    
    def fit(self, x, ways, shots):
        # compute prototypes according to 
        # https://dl.acm.org/doi/10.1145/3449301.3449325
        # Robust Re-weighting Prototypical Networks for Few-Shot Classification 

        x = x.reshape(ways, shots, -1)        
        alph = torch.zeros(ways, x.size(-1)).to(x.device)
        
        for k in range(shots):
            for i in range(shots):
                if i == k:
                    continue
                
                alph[:] += x[:, i]
                
            alph[:] /= (shots - 1)
        
        alph = 1.0 / (x - alph.unsqueeze(1) + 1e-8).pow(2)
        
        return (alph * x).sum(dim = 1) / alph.sum(dim = 1)
    
    def forward(self, x):
        # embedding
        return self.encoder(x).view(x.size(0), -1)