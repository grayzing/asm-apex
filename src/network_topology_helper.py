import numpy as np
from abc import ABC, abstractmethod

from .device_manager import DeviceManager
from .base_station_manager import BaseStationManager
from .sector_manager import SectorManager


class NetworkTopologyHelper(ABC):
    def __init__(self, intersite_distance_meters: float = 500.0, num_rings: int = 2) -> None:
        self.intersite_distance_meters = float(intersite_distance_meters)
        self.num_rings = int(num_rings)

    @abstractmethod
    def setup_network(
        self,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        sector_manager: SectorManager,
    ) -> None:
        raise NotImplementedError

    def attach_device(
        self,
        device_id: int,
        device_position: np.ndarray,
        device_manager: DeviceManager,
    ) -> None:
        position = np.asarray(device_position, dtype=np.float32)
        device_manager.device_position_matrix[device_id, :] = position

    def remove_device(self, device_id: int, device_manager: DeviceManager) -> None:
        device_manager.device_position_matrix[device_id, :] = 0.0
        if hasattr(device_manager, "device_velocity_matrix"):
            device_manager.device_velocity_matrix[device_id, :] = 0.0

    def _calculate_number_of_sites(self) -> int:
        if self.num_rings < 0:
            raise ValueError("num_rings must be non-negative")
        return 1 + 3 * self.num_rings * (self.num_rings + 1)


class HexagonalNetworkTopologyHelper(NetworkTopologyHelper):
    def __init__(self, intersite_distance_meters: float = 500.0, num_rings: int = 2) -> None:
        super().__init__(intersite_distance_meters=intersite_distance_meters, num_rings=num_rings)

    def setup_network(
        self,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        sector_manager: SectorManager,
    ) -> None:
        num_sites = self._calculate_number_of_sites()
        num_sectors = num_sites * 3

        if base_station_manager.base_station_position_matrix.shape[0] < num_sites:
            raise ValueError(
                f"BaseStationManager must support at least {num_sites} sites"
            )
        if sector_manager.sector_parent_base_station_vector.shape[0] < num_sectors:
            raise ValueError(
                f"SectorManager must support at least {num_sectors} sectors"
            )
        if base_station_manager.base_station_sector_adjacency_matrix.shape != (
            sector_manager.sector_parent_base_station_vector.shape[0],
            base_station_manager.base_station_position_matrix.shape[0],
        ):
            raise ValueError(
                "BaseStationManager and SectorManager adjacency dimensions do not match"
            )

        site_positions = self._generate_hexagonal_grid_positions(num_sites)

        base_station_manager.base_station_position_matrix[:num_sites, :] = site_positions
        sector_manager.sector_parent_base_station_vector[:num_sectors] = np.repeat(
            np.arange(num_sites, dtype=np.int16), 3
        )

        adjacency = np.zeros_like(base_station_manager.base_station_sector_adjacency_matrix, dtype=np.int16)
        adjacency[:num_sectors, :num_sites] = 0
        for sector_idx, site_idx in enumerate(sector_manager.sector_parent_base_station_vector[:num_sectors]):
            adjacency[sector_idx, site_idx] = 1
        base_station_manager.base_station_sector_adjacency_matrix[:, :] = adjacency

        if sector_manager.sector_azimuth_angle_deg_matrix.shape[0] >= num_sectors:
            sector_manager.sector_azimuth_angle_deg_matrix[:num_sectors] = np.resize(
                np.array([30.0, 150.0, 270.0], dtype=np.float32), num_sectors
            )

    def _generate_hexagonal_grid_positions(self, num_sites: int) -> np.ndarray:
        angles = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], dtype=np.float32)
        positions = [np.array([0.0, 0.0], dtype=np.float32)]
        ring = 1
        while len(positions) < num_sites:
            for direction in range(6):
                start_angle = np.deg2rad(angles[direction])
                for step in range(ring):
                    if len(positions) >= num_sites:
                        break
                    x = self.intersite_distance_meters * (ring * np.cos(start_angle) - step * np.sin(start_angle) / 2.0)
                    y = self.intersite_distance_meters * (ring * np.sin(start_angle) + step * np.cos(start_angle) / 2.0)
                    positions.append(np.array([x, y], dtype=np.float32))
                if len(positions) >= num_sites:
                    break
            ring += 1
        positions = np.array(positions[:num_sites], dtype=np.float32)
        heights = np.zeros((positions.shape[0], 1), dtype=np.float32)
        return np.concatenate([positions, heights], axis=1)
