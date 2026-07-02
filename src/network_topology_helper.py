import numpy as np

from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager

from abc import ABC, abstractmethod

class NetworkTopologyHelper(ABC):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int) -> None:

        self.base_station_manager = BaseStationManager(num_base_stations=num_base_stations)
        self.sector_manager = SectorManager(num_sectors=num_sectors_per_base_station * num_base_stations)
        self.device_manager = DeviceManager(num_devices=num_devices)

        self.num_sectors = num_sectors_per_base_station * num_base_stations

    @abstractmethod
    def generate_base_station_positions(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def generate_device_positions(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def intialize_network_topology(self):
        self.generate_base_station_positions()
        self.generate_device_positions()

class HexagonalNetworkTopologyHelperWithRandomDevicePlacements(NetworkTopologyHelper):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 500.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices)
        self.inter_site_distance_m = inter_site_distance_m
        self.rng = np.random.default_rng(seed)

    def generate_base_station_positions(self):
        # Generate hexagonal grid positions for base stations
        base_station_positions = []
        for i in range(self.base_station_manager.base_station_position_matrix.shape[0]):
            row = i // 3
            col = i % 3
            x_offset = (col * self.inter_site_distance_m) + (row % 2) * (self.inter_site_distance_m / 2)
            y_offset = row * (self.inter_site_distance_m * np.sqrt(3) / 2)
            base_station_positions.append([x_offset, y_offset, 0.0])  # Assuming z=0 for ground level
        self.base_station_manager.base_station_position_matrix = np.array(base_station_positions, dtype=np.float16)

    def generate_device_positions(self):
        # Randomly place devices within the area covered by the base stations
        x_min = np.min(self.base_station_manager.base_station_position_matrix[:, 0])
        x_max = np.max(self.base_station_manager.base_station_position_matrix[:, 0]) + self.inter_site_distance_m
        y_min = np.min(self.base_station_manager.base_station_position_matrix[:, 1])
        y_max = np.max(self.base_station_manager.base_station_position_matrix[:, 1]) + self.inter_site_distance_m

        device_positions = self.rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.device_manager.device_position_matrix.shape[0], 2))
        device_positions = np.hstack((device_positions, np.zeros((device_positions.shape[0], 1))))  # Assuming z=0 for ground level
        self.device_manager.device_position_matrix = device_positions.astype(np.float16)

    