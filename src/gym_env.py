from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

class HeterogenousNetworkMultiAgentSleepEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(0,31)]

