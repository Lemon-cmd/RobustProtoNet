## PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from einops import rearrange

class ProtoNet(nn.Module):
    # source: https://github.com/tristandeleu/pytorch-meta/blob/master/examples/protonet/model.py
    def __init__(self, in_channels, out_dim, hid_dim = 64):
        super(ProtoNet, self).__init__()
        
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
        # compute prototypes
        return x.reshape(ways, shots, -1).mean(dim = 1)
    
    def forward(self, x):
        # embedding
        return self.encoder(x).view(x.size(0), -1)