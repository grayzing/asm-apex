import time

import numpy as np

from geometry_helper import GeometryHelper
from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from mobility_helper import RandomWalkMobilityHelper
from network_topology_helper import HexagonalNetworkTopologyHelperWithRandomDevicePlacements
from scheduler import QueueAwareProportionalFairPhysicalResourceBlockScheduler
from traffic_generator import TrafficGenerator, ParetoDistributionTrafficGenerator

class Simulator:
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        self.num_base_stations = num_base_stations
        self.num_devices = num_devices
        self.simulation_length_ms = simulation_length_ms

        self.network_topology_helper = HexagonalNetworkTopologyHelperWithRandomDevicePlacements(num_base_stations=self.num_base_stations, num_sectors_per_base_station=3, num_devices=self.num_devices, seed=seed)
        self.network_topology_helper.intialize_network_topology()

        self.base_station_manager = self.network_topology_helper.base_station_manager
        self.sector_manager = self.network_topology_helper.sector_manager
        self.device_manager = self.network_topology_helper.device_manager
        self.geometry_helper = GeometryHelper(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.radio_channel_model = RadioChannelModel(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices, seed=seed)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.mobility_helper = RandomWalkMobilityHelper()
        self.traffic_generator = ParetoDistributionTrafficGenerator(num_devices=self.num_devices, window_length=self.simulation_length_ms, seed=seed)
        self.traffic_generator.generate_device_downlink_bits_matrix()

        self.scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)

    def run_simulation(self):
        total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)
        total_prbs_allocated_per_device = np.zeros(self.num_devices, dtype=np.int64)

        start_time = time.time()
        for step in range(0, self.simulation_length_ms):
            if step % 10 == 0:
                self.mobility_helper.step_devices(self.device_manager)
                self.geometry_helper.update_distance_matrix(self.device_manager, self.sector_manager, self.base_station_manager)
                self.geometry_helper.update_relative_azimuth_angle_deg_matrix(self.device_manager, self.sector_manager, self.base_station_manager)
                self.geometry_helper.update_relative_zenith_angle_deg_matrix(self.device_manager, self.sector_manager, self.base_station_manager)

            self.radio_channel_model.update_path_loss_matrix(self.geometry_helper, self.sector_manager, self.base_station_manager, self.device_manager)
            self.radio_channel_model.update_directional_gain_matrix(self.geometry_helper, self.sector_manager, self.device_manager)
            self.radio_channel_model.update_received_power_matrix_per_resource_element(self.sector_manager)
            self.radio_channel_model.update_sinr_dbm_matrix_per_slot(self.sector_manager)
            self.radio_channel_model.update_spectral_efficiency_matrix()

            self.handover_manager.handover(self.sector_manager, self.device_manager, self.radio_channel_model)
            prb_allocation_matrix, transmitted_bits_per_device = self.scheduler.schedule(
                self.sector_manager,
                self.radio_channel_model,
                self.traffic_generator,
                self.handover_manager,
                self.device_manager,
                step,
            )

            total_transmitted_bits_per_device += transmitted_bits_per_device
            total_prbs_allocated_per_device += np.sum(prb_allocation_matrix, axis=0)

        end_time = time.time()
        wall_clock_seconds = end_time - start_time
        simulated_seconds = self.simulation_length_ms / 1000.0
        device_throughput_mbps = total_transmitted_bits_per_device / simulated_seconds / 1e6
        total_throughput_mbps = np.sum(total_transmitted_bits_per_device) / simulated_seconds / 1e6
        average_throughput_mbps = np.mean(device_throughput_mbps)
        percentile_10_throughput_mbps = np.percentile(device_throughput_mbps, 10)

        print(f"Simulation completed in {wall_clock_seconds:.2f} seconds wall clock.")
        print(f"Simulated duration: {simulated_seconds:.3f} seconds.")
        print(f"Total throughput: {total_throughput_mbps:.3f} Mbps")
        print(f"Average per-device throughput: {average_throughput_mbps:.3f} Mbps")
        print(f"10th percentile per-device throughput: {percentile_10_throughput_mbps:.3f} Mbps")
        print(f"Total PRBs allocated across simulation: {np.sum(total_prbs_allocated_per_device)}")
        print(f"Average PRBs allocated per device: {np.mean(total_prbs_allocated_per_device):.2f}")
        print(f"Median PRBs allocated per device: {np.median(total_prbs_allocated_per_device):.2f}")

if __name__ == "__main__":
    num_base_stations = 61
    num_devices = 1000
    simulation_length_ms = 2500

    simulator = Simulator(num_base_stations=num_base_stations, num_devices=num_devices, simulation_length_ms=simulation_length_ms)
    simulator.run_simulation()