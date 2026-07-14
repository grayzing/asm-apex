import torch
from torch import nn

class Q(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.fc = nn.Sequential(
            nn.Linear(3636, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(256, 1)
        
        self.advantage_stream = nn.Linear(256, 12)

    def forward(self, x):
        # Dueling network
        features = self.fc(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        if advantage.dim() == 1:
            advantage = advantage.unsqueeze(0)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class MixingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sum(x, dim=1).reshape(x.shape[0], 1)
