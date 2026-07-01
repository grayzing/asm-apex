import numpy as np

class BaseStationManager:
    def __init__(self, num_base_stations: int) -> None:
        self.base_station_position_matrix: np.ndarray = np.zeros((num_base_stations, 3), dtype=np.float16)

    def get_all_positions(self) -> np.ndarray:
        return self.base_station_position_matrix