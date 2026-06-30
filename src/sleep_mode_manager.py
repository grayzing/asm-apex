import numpy as np


class SleepModeManager:
    def __init__(self, num_sectors: int) -> None:
        self.sector_sleep_mode_vector: np.ndarray = np.zeros((num_sectors,), dtype=np.int8)
        self.sector_sleep_mode_transition_counter: np.ndarray = np.zeros((num_sectors,), dtype=np.int32)

    def set_sleep_mode(self, sector_id: int, sleep_mode: int) -> None:
        sector_id = int(sector_id)
        sleep_mode = int(sleep_mode)

        current = int(self.sector_sleep_mode_vector[sector_id])
        if current == sleep_mode:
            return

        if sleep_mode in (1, 2):
            if current == 3 and self.sector_sleep_mode_transition_counter[sector_id] > 0:
                self.sector_sleep_mode_transition_counter[sector_id] = 20
            else:
                self.sector_sleep_mode_transition_counter[sector_id] = 0
            self.sector_sleep_mode_vector[sector_id] = sleep_mode
            return

        if sleep_mode == 3:
            self.sector_sleep_mode_transition_counter[sector_id] = 20
            self.sector_sleep_mode_vector[sector_id] = sleep_mode
            return

        self.sector_sleep_mode_vector[sector_id] = sleep_mode
        self.sector_sleep_mode_transition_counter[sector_id] = 0

    def decrement_counters(self) -> None:
        active = self.sector_sleep_mode_transition_counter > 0
        self.sector_sleep_mode_transition_counter[active] -= 1

    def can_change_state(self, sector_id: int) -> bool:
        return self.sector_sleep_mode_transition_counter[int(sector_id)] == 0
