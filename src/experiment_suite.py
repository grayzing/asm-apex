from simulator import Simulator

import time
import gc

import numpy as np

from geometry_helper import GeometryHelper
from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from mobility_helper import RandomWalkMobilityHelper, LinearWalkMobilityHelper
from network_topology_helper import HexagonalNetworkTopologyHelperWithRandomDevicePlacements, HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements
from scheduler import QueueAwareProportionalFairPhysicalResourceBlockScheduler
from traffic_generator import TrafficGenerator, BurstyTrafficGenerator
from sleep_mode_manager import SleepModeManager
from simulation_kpis_handler import SimulationKPIHandler

class EnhancedSleepSimulator(Simulator):
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        super().__init__(num_base_stations = num_base_stations, num_devices = num_devices, simulation_length_ms= simulation_length_ms, seed = seed)

        self.kpis = []

    def reset(self, seed):
        super().reset(seed)

    def sleep_xapp(self, curr_step):
        # Put a random sector to sleep
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
                        #print(first_sector)
                        
                        self.handover_manager.manual_handover(first_sector, device, self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0):
                #print(f"No devices attached to sector{sector}, putting to sleep")
                #print(self.handover_manager.sector_device_association_matrix)
                self.sleep_mode_manager.set_sleep_mode(
                    sector_id=sector,
                    sleep_mode=1,
                    sector_manager=self.sector_manager
                )
    def step(self, step_number):
        super().step(step_number)
        self.kpis.append((
            step_number,
            self.kpi_handler.calculate_average_throughput_mbps(step_number),
            self.kpi_handler.calculate_throughput_percentile(10,step_number)
        ))
    def run_simulation(self):
        super().run_simulation()

class BasicSleepSimulator(Simulator):
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        super().__init__(num_base_stations = num_base_stations, num_devices = num_devices, simulation_length_ms= simulation_length_ms, seed = seed)

    def reset(self, seed):
        super().reset(seed)

    def sleep_xapp(self, curr_step):
        # Put a random sector to sleep
        for sector in range(self.num_sectors):
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0) and self.sleep_mode_manager.sector_sleep_mode_matrix[sector] == 0:
                #print(f"No devices attached to sector{sector}, putting to sleep")
                #print(self.handover_manager.sector_device_association_matrix)
                self.sleep_mode_manager.set_sleep_mode(
                    sector_id=sector,
                    sleep_mode=1,
                    sector_manager=self.sector_manager
                )
    def step(self, step_number):
        super().step(step_number)
    def run_simulation(self):
        super().run_simulation()

class RandomSleepSimulator(Simulator):
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        super().__init__(num_base_stations = num_base_stations, num_devices = num_devices, simulation_length_ms= simulation_length_ms, seed = seed)

    def reset(self, seed):
        super().reset(seed)

    def sleep_xapp(self, curr_step):
        # Put a random sector to sleep
        self.sleep_mode_manager.set_sleep_mode(
                    sector_id=self.rng.integers(0,self.num_sectors),
                    sleep_mode=self.rng.integers(0,4),
                    sector_manager=self.sector_manager
                )
                
    def step(self, step_number):
        super().step(step_number)
    def run_simulation(self):
        super().run_simulation()

if __name__ == "__main__":
    poop=209432
    sim = EnhancedSleepSimulator(19, 500, 500, seed=poop)
    sim.run_simulation()
    sim.kpi_handler.print_kpis(500)
    print(np.count_nonzero(sim.sleep_mode_manager.sector_sleep_mode_matrix))

    sim1 = BasicSleepSimulator(19,500,500,seed=poop)
    sim1.run_simulation()
    sim1.kpi_handler.print_kpis(500)
    print(np.count_nonzero(sim1.sleep_mode_manager.sector_sleep_mode_matrix))

    sim2 = Simulator(19,500,500,seed=poop)
    sim2.run_simulation()
    sim2.kpi_handler.print_kpis(500)
    print(np.count_nonzero(sim2.sleep_mode_manager.sector_sleep_mode_matrix))

    sim3 = RandomSleepSimulator(19,500,500,seed=poop)
    sim3.run_simulation()
    sim3.kpi_handler.print_kpis(500)
    print(np.count_nonzero(sim3.sleep_mode_manager.sector_sleep_mode_matrix))
    