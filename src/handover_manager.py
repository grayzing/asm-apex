import numpy as np

from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from sleep_mode_manager import SleepModeManager

from abc import ABC, abstractmethod

class HandoverManager(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.sector_device_association_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.int8) # where [i][j] = 1 if sector i is giving downlink data to device j

    def get_serving_sector_indices(self) -> np.ndarray:
        return np.argmax(self.sector_device_association_matrix, axis=0)

    @abstractmethod
    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel, sleep_mode_manager: SleepModeManager):
        raise NotImplementedError("Subclasses must implement this method.")
    
class RSRPBasedHandoverManager(HandoverManager):
    def __init__(self, num_sectors: int, num_devices: int, hysterisis: float = 3.0) -> None:
        super().__init__(num_sectors, num_devices)
        self.hysterisis = hysterisis

    def get_serving_sector_indices(self) -> np.ndarray:
        return np.argmax(self.sector_device_association_matrix, axis=0)

    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel, sleep_mode_manager: SleepModeManager):
        # RSRP based handover, ignoring sectors who are asleep
        sector_sleep_mode_indices = sleep_mode_manager.get_sector_sleep_mode_indices()
        modified_radio_channel_model: np.ndarray = np.copy(radio_channel_model.received_power_dbm_matrix_per_resource_element)
        modified_radio_channel_model[sector_sleep_mode_indices] = np.nan
        if np.isnan(modified_radio_channel_model).all():
            # Prevent ALL sectors from being switched off.
            # So turn on the center
            modified_radio_channel_model[0:3] = radio_channel_model.received_power_dbm_matrix_per_resource_element[0:3]
        best_sector_indices: np.ndarray = np.nanargmax(radio_channel_model.received_power_dbm_matrix_per_resource_element, axis=0) 
        self.sector_device_association_matrix = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.sector_device_association_matrix[best_sector_indices, np.arange(self.num_devices)] = 1