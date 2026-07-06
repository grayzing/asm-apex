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
        

        self.sector_manager.sector_parent_base_station_vector = np.repeat(
            np.arange(num_base_stations, dtype=np.int16),
            num_sectors_per_base_station,
        )

        reuse_frequencies_ghz = np.array([6.0, 6.5, 7.0], dtype=np.float32)
        base_station_center_freqs = reuse_frequencies_ghz[np.arange(num_base_stations) % reuse_frequencies_ghz.shape[0]]
        self.sector_manager.center_freq_ghz_matrix = np.repeat(base_station_center_freqs, num_sectors_per_base_station)

        self.num_sectors = num_sectors_per_base_station * num_base_stations
        self.num_base_stations = num_base_stations
        self.num_sectors_per_base_station = num_sectors_per_base_station
        self.num_devices = num_devices

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
        """
        Generates a true mathematical hexagonal grid spiraling outwards in rings.
        This completely eliminates row/column box alignment issues.
        """
        base_station_positions = []
        
        base_station_positions.append([0.0, 0.0, 10.0]) # Z=10 default for micros
        
        axial_coords = []
        
        for ring in [1, 2]:
            q = ring
            r = 0
            for direction in range(6):
                for step in range(ring):
                    axial_coords.append((q, r))
                    if direction == 0: r -= 1
                    elif direction == 1: q -= 1
                    elif direction == 2: q -= 1; r += 1
                    elif direction == 3: r += 1
                    elif direction == 4: q += 1
                    elif direction == 5: q += 1; r -= 1
        
        for q, r in axial_coords:
            if len(base_station_positions) >= self.base_station_manager.base_station_position_matrix.shape[0]:
                break
            x = self.inter_site_distance_m * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
            y = self.inter_site_distance_m * (3.0/2 * r)
            base_station_positions.append([x, y, 10.0]) # Default Z=10 for micros

        self.base_station_manager.base_station_position_matrix = np.array(base_station_positions[:self.num_base_stations], dtype=np.float32)

    def generate_device_positions(self):
        x_min = np.min(self.base_station_manager.base_station_position_matrix[:, 0])
        x_max = np.max(self.base_station_manager.base_station_position_matrix[:, 0])
        y_min = np.min(self.base_station_manager.base_station_position_matrix[:, 1])
        y_max = np.max(self.base_station_manager.base_station_position_matrix[:, 1])

        velocity_multipliers = self.rng.integers(-1, 2, size=(self.num_devices, 3))
        self.device_manager.device_velocity_matrix *= velocity_multipliers

        device_positions = self.rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.device_manager.device_position_matrix.shape[0], 2))
        device_positions = np.hstack((device_positions, np.zeros((device_positions.shape[0], 1))))  
        self.device_manager.device_position_matrix = device_positions.astype(np.float32)


class HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements(HexagonalNetworkTopologyHelperWithRandomDevicePlacements):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 200.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices, inter_site_distance_m=inter_site_distance_m, seed=seed)
        self.reuse_frequencies_ghz = np.array([35.0, 37.0, 39.0], dtype=np.float32)
        self.sector_manager.tx_power_dbm_matrix = np.full((self.sector_manager.num_sectors, ), 35.0, dtype=np.float32) 
        self.sector_manager.bandwidth_mhz_matrix = np.full((self.sector_manager.num_sectors, ), 100.0, dtype=np.float32) 
        self.sector_manager.sector_numerology_matrix = np.full((self.sector_manager.num_sectors, ), 3, dtype=np.int8) 

    def generate_base_station_positions(self):
        super().generate_base_station_positions()
        
        base_station_center_freqs = self.reuse_frequencies_ghz[np.arange(self.base_station_manager.base_station_position_matrix.shape[0]) % self.reuse_frequencies_ghz.shape[0]]
        self.sector_manager.center_freq_ghz_matrix = np.repeat(base_station_center_freqs, self.num_sectors_per_base_station)

        self.base_station_manager.base_station_position_matrix[0] = [0.0, 0.0, 25.0]
        
        self.sector_manager.center_freq_ghz_matrix[0:3] = [6.0, 6.5, 7.0]  
        self.sector_manager.tx_power_dbm_matrix[0:3] = 49.0
        self.sector_manager.bandwidth_mhz_matrix[0:3] = 20.0 
        self.sector_manager.sector_numerology_matrix[0:3] = 1