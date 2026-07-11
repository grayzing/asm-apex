import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
from simulator import Simulator

# --- Simulator Classes with Step-by-Step Logging ---

class BaseSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleeping_history = []
    
    def step(self, step_number):
        super().step(step_number)
        # Record number of sectors in a sleep mode (> 0)
        num_sleeping = np.count_nonzero(self.sleep_mode_manager.sector_sleep_mode_matrix)
        self.sleeping_history.append(num_sleeping)

class EnhancedSleepSimulator(BaseSimulator):
    # (Existing sleep_xapp logic remains the same)
    def sleep_xapp(self, curr_step):
        for sector in range(self.num_sectors):
            if self.sleep_mode_manager.sector_sleep_mode_matrix[sector] == 0:
                attached_devices_indices = np.where(self.handover_manager.sector_device_association_matrix[sector,:] == 0)
                average_throughput_of_sector = self.kpi_handler.calculate_throughput_mbps(curr_step)[attached_devices_indices].mean()
                if self.sector_manager.sector_physical_resource_block_utilization[sector] < 0.5 and average_throughput_of_sector < 20.0:
                    for device in attached_devices_indices:
                        rsrp_device_slice = self.radio_channel_model.received_power_dbm_matrix_per_resource_element[:,device]
                        prb_utilization = self.sector_manager.sector_physical_resource_block_utilization[:,np.newaxis]
                        rsrp_device_mask = (rsrp_device_slice >= -90) & (prb_utilization <= 0.9)
                        rsrp_device_mask[sector, :] = False
                        first_sector = np.argmax(np.any(rsrp_device_mask, axis=1))
                        self.handover_manager.manual_handover(first_sector, device, self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0):
                self.sleep_mode_manager.set_sleep_mode(sector_id=sector, sleep_mode=1, sector_manager=self.sector_manager)

class BasicSleepSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        for sector in range(self.num_sectors):
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0) and self.sleep_mode_manager.sector_sleep_mode_matrix[sector] == 0:
                self.sleep_mode_manager.set_sleep_mode(sector_id=sector, sleep_mode=1, sector_manager=self.sector_manager)

class RandomSleepSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        self.sleep_mode_manager.set_sleep_mode(sector_id=self.rng.integers(0,self.num_sectors), sleep_mode=self.rng.integers(0,4), sector_manager=self.sector_manager)

class VDNSleepSimulator(BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = torch.load("q_net.pth", weights_only=False, map_location=torch.device('cpu'))

    def sleep_xapp(self, curr_step):
        for agent in range(0,31):
            obs = self.take_observation(agent, self)
            action = int(torch.argmax(self.q_net(obs)))
            sector_id = int(agent * 3 + (action // 4))
            self.sleep_mode_manager.set_sleep_mode(sector_id=sector_id, sleep_mode=int(action % 4), sector_manager=self.sector_manager)

    def take_observation(self, agent_id, simulator):
        return torch.from_numpy(np.concatenate([np.stack([simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]], simulator.radio_channel_model.sinr_dbm_matrix_per_slot[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]], axis=0).flatten(), simulator.sleep_mode_manager.sector_sleep_mode_matrix[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]])).to(torch.float32)

import matplotlib.pyplot as plt

# --- Base Class for Tracking ---
class BaseTrackingSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleeping_history = []
    
    def step(self, step_number):
        super().step(step_number)
        # Count non-zero sleep modes as sleeping sectors
        num_sleeping = np.count_nonzero(self.sleep_mode_manager.sector_sleep_mode_matrix)
        self.sleeping_history.append(num_sleeping)

# Inherit simulators from BaseTrackingSimulator (keep your existing logic inside)
class EnhancedSleepSimulator(BaseTrackingSimulator, EnhancedSleepSimulator): pass
class BasicSleepSimulator(BaseTrackingSimulator, BasicSleepSimulator): pass
class RandomSleepSimulator(BaseTrackingSimulator, RandomSleepSimulator): pass
class VDNSleepSimulator(BaseTrackingSimulator, VDNSleepSimulator): pass
class NormalSimulator(BaseTrackingSimulator, Simulator): pass

# --- Batch Execution & Export ---
def run_experiment(SimulatorClass, method_name, n=20):
    all_data = []
    for i in range(n):
        sim = SimulatorClass(31, 500, 500, seed=1000+i)
        sim.run_simulation()
        
        tp = sim.kpi_handler.calculate_throughput_mbps(500)
        all_data.append({
            'Method': method_name,
            'Trial': i,
            'P10_Throughput': np.percentile(tp, 10),
            'Avg_Throughput': np.mean(tp),
            'Avg_Sleeping_Sectors': np.mean(sim.sleeping_history)
        })
        del sim; gc.collect()
    return all_data

methods = {
    "Random": RandomSleepSimulator, 
    "Normal": NormalSimulator, 
    "Enhanced": EnhancedSleepSimulator, 
    "Basic": BasicSleepSimulator, 
    "VDN": VDNSleepSimulator
}

results = []
for name, cls in methods.items():
    print(f"Running 20 simulations for {name}...")
    results.extend(run_experiment(cls, name))

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("simulation_results.csv", index=False)
print("Data saved to simulation_results.csv")

# --- Accessible Visualization ---
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
styles = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'D'), ('-', 'x')]

for (name, cls), (ls, marker) in zip(methods.items(), styles):
    subset = df[df['Method'] == name]
    for i, col in enumerate(['P10_Throughput', 'Avg_Throughput']):
        data = np.sort(subset[col].values)
        axs[i].plot(data, np.linspace(0, 1, len(data)), label=name, ls=ls, marker=marker, markevery=2)

axs[0].set_title("CDF: 10th Percentile Throughput"); axs[1].set_title("CDF: Average Throughput")
[ax.legend() or ax.grid(True) for ax in axs]

plt.figure(figsize=(10, 5))
bp = plt.boxplot([df[df['Method']==m]['Avg_Sleeping_Sectors'] for m in methods.keys()], 
                 labels=methods.keys(), patch_artist=True)
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_hatch('//')

plt.title("Average Number of Sleeping Sectors per Method")
plt.show()