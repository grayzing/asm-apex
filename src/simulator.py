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

class Simulator:
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        self.num_base_stations = num_base_stations
        self.num_devices = num_devices
        self.num_sectors = num_base_stations * 3
        self.simulation_length_ms = simulation_length_ms
        self.rng = np.random.default_rng(seed=seed)
        
        self.sleep_mode_manager = SleepModeManager(num_sectors=self.num_base_stations * 3)
        self.network_topology_helper = HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements(num_base_stations=self.num_base_stations, num_sectors_per_base_station=3, num_devices=self.num_devices, seed=seed)
        self.network_topology_helper.intialize_network_topology()

        self.base_station_manager = self.network_topology_helper.base_station_manager
        self.sector_manager = self.network_topology_helper.sector_manager
        self.device_manager = self.network_topology_helper.device_manager
        self.geometry_helper = GeometryHelper(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.radio_channel_model = RadioChannelModel(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices, seed=seed)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.mobility_helper = RandomWalkMobilityHelper()
        self.traffic_generator = BurstyTrafficGenerator(num_devices=self.num_devices, window_length=self.simulation_length_ms, seed=seed)
        self.traffic_generator.generate_device_downlink_bits_matrix()

        self.scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(num_sectors=self.network_topology_helper.num_sectors, num_devices=self.num_devices)
        self.kpi_handler = SimulationKPIHandler(self.num_devices, self.num_sectors)

    def reset(self, seed):
        if seed is None:
            seed = 42
        self.rng = np.random.default_rng(seed=seed)
        self.sleep_mode_manager.sector_sleep_mode_countdown = np.zeros((self.num_sectors, ), dtype=np.int8)
        self.sleep_mode_manager.sector_sleep_mode_matrix = np.zeros((self.num_sectors, ), dtype=np.int32)

        self.network_topology_helper.generate_device_positions()
        self.sector_manager.sector_physical_resource_block_utilization: np.ndarray = np.full((self.num_sectors, ), 0.05, dtype=np.float32)

        self.geometry_helper.relative_azimuth_angle_deg_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.geometry_helper.relative_zenith_angle_deg_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.geometry_helper.distance_matrix_meters_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)

        self.radio_channel_model.path_loss_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.directional_gain_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.received_power_dbm_matrix_per_resource_element: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.sinr_dbm_matrix_per_slot: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.spectral_efficiency_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)

        self.handover_manager.sector_device_association_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.traffic_generator.generate_device_downlink_bits_matrix()

        self.kpi_handler.total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)

        gc.collect()

    def set_random_sleep_mode(self):
        # Put a random sector to sleep
        sector_sleep_id = self.rng.integers(0, self.num_base_stations*3)
        random_sleep_mode = 1
        #print(f"Sector sleep id: {sector_sleep_id}")
        #print(f"Random sleep mode: {random_sleep_mode}")
        self.sleep_mode_manager.set_sleep_mode(
            sector_id=sector_sleep_id,
            sleep_mode=random_sleep_mode,
            sector_manager=self.sector_manager
        )

    def step(self, step_number):
        if step_number % 10 == 0:
            self.mobility_helper.step_devices(self.device_manager)
            self.geometry_helper.update_spatial_geometry(self.device_manager,self.sector_manager,self.base_station_manager)

        self.radio_channel_model.update_path_loss_matrix(self.geometry_helper, self.sector_manager, self.base_station_manager, self.device_manager)
        self.radio_channel_model.update_directional_gain_matrix(self.geometry_helper, self.sector_manager, self.device_manager)
        self.radio_channel_model.update_received_power_matrix_per_resource_element(self.sector_manager)
        self.radio_channel_model.update_sinr_dbm_matrix_per_slot(self.sector_manager)
        self.radio_channel_model.update_spectral_efficiency_matrix(self.sleep_mode_manager)

        self.sleep_mode_manager.tick()
        self.handover_manager.handover(self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)

        prb_allocation_matrix, transmitted_bits_per_device = self.scheduler.schedule(
            self.sector_manager,
            self.radio_channel_model,
            self.traffic_generator,
            self.handover_manager,
            self.device_manager,
            self.sleep_mode_manager,
            step_number,
        )

        self.kpi_handler.update_kpis(transmitted_bits_per_device)

    def run_simulation(self):
        self.kpi_handler.start_clock()
        for step in range(0, self.simulation_length_ms):
            self.step(step_number=step)
            
        self.kpi_handler.end_clock()
        self.kpi_handler.get_elapsed_time()