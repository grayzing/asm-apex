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

import time

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
            self.scheduler.schedule(self.sector_manager, self.radio_channel_model, self.traffic_generator, self.handover_manager, self.device_manager, step)

            print(f"Step {step + 1}/{self.simulation_length_ms} completed.")
            print(f"Device Physical Resource Block Allocation Vector: {self.device_manager.device_physical_resource_block_allocation_vector}")
            print(f"Sector Physical Resource Block Utilization: {self.sector_manager.sector_physical_resource_block_utilization}")

        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    num_base_stations = 19
    num_devices = 50
    simulation_length_ms = 10

    simulator = Simulator(num_base_stations=num_base_stations, num_devices=num_devices, simulation_length_ms=simulation_length_ms)
    simulator.run_simulation()