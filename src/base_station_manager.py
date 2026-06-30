import numpy as np

"""
base_station_sector_adjacency_matrix: np.ndarray
# This is an N x M biadjacency matrix
base_station_position_matrix: np.ndarray
# N x 3 matrix
"""

class BaseStationManager:
    def __init__(self, num_base_stations: int, num_sectors: int) -> None:
        self.base_station_sector_adjacency_matrix: np.ndarray = np.zeros((num_sectors, num_base_stations), dtype=np.int16)
        self.base_station_position_matrix: np.ndarray = np.zeros((num_base_stations, 3), dtype=np.float16)

    def get_all_positions(self) -> np.ndarray:
        return self.base_station_position_matrix