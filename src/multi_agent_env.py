from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
import time

from simulator import Simulator

class UltraDenseHetNetEnvironment(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = [f"agent_{i}" for i in range(0,32)]
        self.agents = self.possible_agents

    def get_observation_space(self, agent_id):
        return gym.spaces.Box(low=-200, high=200, shape=(18000, ), dtype=np.float32)

    def get_action_space(self, agent_id):
        return gym.spaces.Discrete(12)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.num_steps = 0
        if seed==None:
            seed=60626
        self.simulator: Simulator = Simulator(31, 500, 2500, seed=seed)
        self.simulator.step(self.num_steps)
        initial_observation = {
            f"agent_{i}": np.stack(
                [
                    self.simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]],
                    self.simulator.radio_channel_model.sinr_dbm_matrix_per_slot[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]]
                ],
                axis=0
            ).flatten() for i in range(0,32)
        }
        return initial_observation, {}
    def step(self, action_dict):
        # Record observations
        self.num_steps += 1
        agent_observations = {
            f"agent_{i}": np.stack(
                [
                    self.simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]],
                    self.simulator.radio_channel_model.sinr_dbm_matrix_per_slot[self.simulator.sector_manager.neighboring_sectors_indices_matrix[i]]
                ],
                axis=0
            ).flatten() for i in range(0,32)
        }
        # Take actions
        for action, agent_id in enumerate(action_dict):
            base_station_id = int(str.split(agent_id, '_')[-1])
            sector_id = base_station_id + action // 4
            sleep_mode_id = action % 4
            self.simulator.sleep_mode_manager.set_sleep_mode(sector_id, sleep_mode_id, self.simulator.sector_manager)
        self.simulator.step(self.num_steps)
        
        # Reward calculations
        alpha = 1.25
        beta = 0.95

        ratio_of_active_sectors = np.count_nonzero(self.simulator.sleep_mode_manager.get_sector_sleep_mode_indices == 0).mean()
        ratio_of_sm1_sectors = np.count_nonzero(self.simulator.sleep_mode_manager.get_sector_sleep_mode_indices == 1).mean()
        ratio_of_sm2_sectors = np.count_nonzero(self.simulator.sleep_mode_manager.get_sector_sleep_mode_indices == 2).mean()
        ratio_of_sm3_sectors = np.count_nonzero(self.simulator.sleep_mode_manager.get_sector_sleep_mode_indices == 3).mean()

        ratio_of_sla_acceptable_devices = np.count_nonzero(self.simulator.kpi_handler.calculate_throughput_mbps(self.num_steps) >= 3.0).mean()

        reward_sleep = alpha * -3 * ratio_of_active_sectors + ratio_of_sm1_sectors + 3 * ratio_of_sm2_sectors + 6 * ratio_of_sm3_sectors

        reward_sla = beta * 6 * ratio_of_sla_acceptable_devices


        reward_dict = {
            f"agent_{i}": reward_sleep + reward_sla for i in range(0,32)
        }

        truncated_dict = {
            "__all__": self.num_steps >= 2499
        }

        terminated_dict = {
            "__all__": False
        }
        # Leave terminated, info dicts empty
        return agent_observations, reward_dict, terminated_dict, truncated_dict, {}

if __name__ == "__main__":
    pass

