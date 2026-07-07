import torch
from torch import nn

class Q(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.fc = nn.Sequential(
            nn.Linear(18018, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.ReLU()
        )
        self.action_out = nn.Linear(512, 12)

    def forward(self, x):
        return self.action_out(self.fc(x))

class MixingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sum(x)
