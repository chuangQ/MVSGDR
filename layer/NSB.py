import torch
import torch.nn as nn
import torch.nn.functional as F


class NSB(nn.Module):

    def __init__(self, channels, dropout=0.1):
        super(NSB, self).__init__()
        self.lin1 = nn.Linear(channels, channels)
        self.lin2 = nn.Linear(channels, channels)
        self.Dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.Dropout(F.relu(self.lin1(x)))
        x = self.Dropout(self.lin2(x))
        return x
