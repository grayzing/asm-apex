from geometry_helper import GeometryHelper
from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from mobility_helper import RandomWalkMobilityHelper

class Simulator:
    def __init__(self, num_sectors: int, num_devices: int, simulation_length_ms: int = 1000) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.simulation_length_ms = simulation_length_ms

        self.base_station_manager = BaseStationManager(num_base_stations=2)
        self.sector_manager = SectorManager(num_sectors=num_sectors)
        self.device_manager = DeviceManager(num_devices=num_devices)
        self.geometry_helper = GeometryHelper(num_sectors=num_sectors, num_devices=num_devices)
        self.radio_channel_model = RadioChannelModel(num_sectors=num_sectors, num_devices=num_devices)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=num_sectors, num_devices=num_devices)
        self.mobility_helper = RandomWalkMobilityHelper(num_devices=num_devices)

    def run_simulation(self):
        for step in range(1, self.simulation_length_ms+1):
            # Update device positions based on mobility model
            self.mobility_helper.step_devices(self.device_manager)
            # Update geometry-related matrices
            self.geometry_helper.update_distance_matrix(self.device_manager, self.sector_manager, self.base_station_manager)
            self.geometry_helper.update_relative_azimuth_angle_deg_matrix(self.device_manager, self.sector_manager, self.base_station_manager)
            self.geometry_helper.update_relative_zenith_angle_deg_matrix(self.device_manager, self.sector_manager, self.base_station_manager)

            # Update radio channel model matrices
            self.radio_channel_model.update_path_loss_matrix(self.geometry_helper, self.sector_manager, self.base_station_manager, self.device_manager)
            self.radio_channel_model.update_directional_gain_matrix(self.geometry_helper, self.sector_manager, self.device_manager)
            self.radio_channel_model.update_received_power_matrix_per_resource_element(self.sector_manager)
            self.radio_channel_model.update_sinr_dbm_matrix_per_slot(self.sector_manager)
            self.radio_channel_model.update_spectral_efficiency_matrix()

            # Perform handover decisions
            self.handover_manager.handover(self.sector_manager, self.device_manager, self.radio_channel_model)