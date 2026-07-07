from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

from simulator import Simulator

class HeterogenousNetworkMultiAgentSleepEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(0,32)]
        self.agents = possible_agents

    def get_observation_space(self, agent_id):
        return gym.spaces.Box(low=-200, high=0, shape=(18, 500, 2), dtype=np.float32)

    def get_action_space(self, agent_id):
        return gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator: Simulator = Simulator(31, 500, 2500, seed=seed)
        initial_observation = {
            f"agent_{i}": np.ndarray(
                [
                    self.simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]],
                    self.simulator.radio_channel_model.sinr_dbm_matrix_per_slot[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]]
                ]
            ) for i in range(0,32)
        }
        return initial_observation, {}
    def step(self, action_dict):
        self.simulator.step()

