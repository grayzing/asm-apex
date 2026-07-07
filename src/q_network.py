import torch
from torch import nn

class Q(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.fc = nn.Sequential(
            nn.Linear(18018, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.action_out = nn.Linear(256, 12)

    def forward(self, x):
        return self.action_out(self.fc(x))

class MixingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sum(x, dim=1).reshape(x.shape[0], 1)
