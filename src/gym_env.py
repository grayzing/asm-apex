from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

from simulator import Simulator

class HeterogenousNetworkMultiAgentSleepEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(0,31)]
        self.agents = possible_agents

        self.observation_spaces = {a: gym.spaces.Box() for a in self.agents}
        self.action_spaces = {a: gym.spaces.Discrete(4) for a in self.agents}

    def reset(self, *, seed=None, options=None):
        self.simulator: Simulator = Simulator(32, 500, 2500, seed=seed)
        
