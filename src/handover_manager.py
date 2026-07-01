import numpy as np

from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel

from abc import ABC, abstractmethod

class HandoverManager(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.sector_device_association_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.int8) # where [i][j] = 1 if sector i is giving downlink data to device j

    @abstractmethod
    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel):
        raise NotImplementedError("Subclasses must implement this method.")
    
class RSRPBasedHandoverManager(HandoverManager):
    def __init__(self, num_sectors: int, num_devices: int, hysterisis: float = 3.0) -> None:
        super().__init__(num_sectors, num_devices)
        self.hysterisis = hysterisis

    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel):
        best_sector_indices: np.ndarray = np.argmax(radio_channel_model.received_power_dbm_matrix_per_resource_element + self.hysterisis, axis=0)
        self.sector_device_association_matrix = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.sector_device_association_matrix[best_sector_indices, np.arange(self.num_devices)] = 1