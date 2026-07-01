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
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000) -> None:
        self.num_base_stations = num_base_stations
        self.num_devices = num_devices
        self.simulation_length_ms = simulation_length_ms

        self.network_topology_helper = HexagonalNetworkTopologyHelperWithRandomDevicePlacements(num_base_stations=self.num_base_stations, num_sectors_per_base_station=3, num_devices=self.num_devices)
        self.network_topology_helper.intialize_network_topology()

        self.base_station_manager = self.network_topology_helper.base_station_manager
        self.sector_manager = self.network_topology_helper.sector_manager
        self.device_manager = self.network_topology_helper.device_manager
        self.geometry_helper = GeometryHelper(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.radio_channel_model = RadioChannelModel(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.mobility_helper = RandomWalkMobilityHelper()
        self.traffic_generator = ParetoDistributionTrafficGenerator(num_devices=self.num_devices, window_length=self.simulation_length_ms)
        self.traffic_generator.generate_device_downlink_bits_matrix()

    def run_simulation(self):
        start_time = time.time()
        for step in range(0, self.simulation_length_ms):
            print(f"Step {step}:")
            print("Distance Matrix (meters):")
            print(self.geometry_helper.distance_matrix_meters_matrix)
            print("\nRelative Azimuth Angle Matrix (degrees):")
            print(self.geometry_helper.relative_azimuth_angle_deg_matrix)
            print("\nRelative Zenith Angle Matrix (degrees):")
            print(self.geometry_helper.relative_zenith_angle_deg_matrix)
            print("\nPath Loss Matrix (dB):")
            print(self.radio_channel_model.path_loss_matrix)
            print("\nDirectional Gain Matrix (dB):")
            print(self.radio_channel_model.directional_gain_matrix)
            print("\nReceived Power Matrix per Resource Element (dBm):")
            print(self.radio_channel_model.received_power_dbm_matrix_per_resource_element)
            print("\nSINR Matrix per Slot (dB):")
            print(self.radio_channel_model.sinr_dbm_matrix_per_slot)
            print("\nSpectral Efficiency Matrix (bps/Hz):")
            print(self.radio_channel_model.spectral_efficiency_matrix)
            print("\nDevice Downlink Bits Matrix:")
            print(self.traffic_generator.device_downlink_bits_matrix)
            print("\nDevice Physical Resource Block Allocation Vector:")
            print(self.device_manager.device_physical_resource_block_allocation_vector)
            print("\nSector Physical Resource Block Utilization:")
            print(self.sector_manager.sector_physical_resource_block_utilization)
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

            scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
            scheduler.schedule(self.sector_manager, self.radio_channel_model, self.traffic_generator, self.handover_manager, self.device_manager, step)

        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    num_base_stations = 19
    num_devices = 25
    simulation_length_ms = 500

    simulator = Simulator(num_base_stations=num_base_stations, num_devices=num_devices, simulation_length_ms=simulation_length_ms)
    simulator.run_simulation()