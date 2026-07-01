import numpy as np
from device_manager import DeviceManager
from sector_manager import SectorManager
from base_station_manager import BaseStationManager
from scipy.spatial.distance import cdist

class GeometryHelper:
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.relative_azimuth_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.relative_zenith_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.distance_matrix_meters_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)

    def update_distance_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager):
        device_positions: np.ndarray = device_manager.get_all_positions() # U x 3
        sector_positions: np.ndarray = base_station_manager.get_all_positions()[sector_manager.sector_parent_base_station_vector] # M x 3
        self.distance_matrix_meters_matrix = cdist(sector_positions, device_positions).astype(np.float32) + 1e-9 # M x U
        # self.distance_matrix_meters_matrix[i][j] = distance between sector i and device j

    def update_relative_azimuth_angle_deg_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager):
        device_positions: np.ndarray = device_manager.get_all_positions() # U x 3
        sector_positions: np.ndarray = base_station_manager.get_all_positions()[sector_manager.sector_parent_base_station_vector] # M x 3
        
        displacement_matrix_meters: np.ndarray = device_positions[np.newaxis, :, :] - sector_positions[:, np.newaxis, :] # M x U x 3
        dx_meters: np.ndarray = displacement_matrix_meters[:, :, 0]
        dy_meters: np.ndarray = displacement_matrix_meters[:, :, 1]

        azimuth_abs = np.degrees(np.arctan2(dy_meters, dx_meters))

        azimuth_raw = azimuth_abs - sector_manager.sector_azimuth_angle_deg_matrix[:, np.newaxis]
        self.relative_azimuth_angle_deg_matrix = (azimuth_raw + 180) % 360 - 180

    def update_relative_zenith_angle_deg_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager):
        device_positions: np.ndarray = device_manager.get_all_positions() # U x 3
        sector_positions: np.ndarray = base_station_manager.get_all_positions()[sector_manager.sector_parent_base_station_vector] # M x 3
        
        displacement_matrix_meters: np.ndarray = device_positions[np.newaxis, :, :] - sector_positions[:, np.newaxis, :] # M x U x 3
        dz_meters: np.ndarray = displacement_matrix_meters[:, :, 2] # M x U
        cosine_zenith = np.clip(dz_meters / self.distance_matrix_meters_matrix, -1.0, 1.0)
        absolute_zenith_deg = np.clip(np.degrees(np.arccos(cosine_zenith)), 0.0, 180.0)
        
        sector_downtilt = sector_manager.downtilt_deg_matrix[:, np.newaxis]
        self.relative_zenith_angle_deg_matrix = absolute_zenith_deg - (90.0 + sector_downtilt)
