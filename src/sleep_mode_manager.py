import numpy as np
from sector_manager import SectorManager

class SleepModeManager:
    def __init__(self, num_sectors: int):
        self.num_sectors = num_sectors
        self.sector_sleep_mode_matrix = np.zeros((self.num_sectors, ), dtype=np.int8)
        self.sector_sleep_mode_countdown = np.zeros((self.num_sectors, ), dtype=np.int32)
        # 0 - ACTIVE, 1 - SM1, 2 - SM2, 3 - SM3

    def tick(self):
        self.sector_sleep_mode_countdown -= 1
        self.sector_sleep_mode_countdown = np.maximum(self.sector_sleep_mode_countdown, 0)

    def set_sleep_mode(self, sector_id: int, sleep_mode: int, sector_manager: SectorManager) -> None:
        assert sleep_mode in [0,1,2,3], "Invalid sleep mode"
        if sleep_mode == 3:
            self.sector_sleep_mode_matrix[sector_id] = sleep_mode
            self.sector_sleep_mode_countdown[sector_id] = 5 + 10 # activation and minimum time in sleep mode
            return
        if self.sector_sleep_mode_countdown[sector_id] == 0:
            self.sector_sleep_mode_matrix[sector_id] = sleep_mode


    def get_sector_sleep_mode_indices(self) -> np.ndarray:
        sector_sleep_mode_indices: np.ndarray = np.where(self.sector_sleep_mode_matrix > 0)
        return sector_sleep_mode_indices
