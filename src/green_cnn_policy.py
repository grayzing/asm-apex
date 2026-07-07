import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class GreenCNNPolicy(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.fc = nn.Sequential(
            nn.Linear(obs_space.shape[0], 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.LSTM(4096, 2048, 1),
            nn.ReLU(),
            nn.Linear(2048,512),
            nn.ReLU()
        )
        self.action_out = nn.Linear(512, num_outputs)
        self.value_out = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        x = self.fc(input_dict["obs"].float())
        return self.action_out(x), state

    def value_function(self):
        return self.value_out(self._last_features)

