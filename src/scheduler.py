import numpy as np

from sector_manager import SectorManager
from radio_channel_model import RadioChannelModel
from traffic_generator import TrafficGenerator

from abc import ABC, abstractmethod

class PhysicalResourceBlockScheduler(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

    @abstractmethod
    def schedule(self, sector_manager: SectorManager, radio_channel_model: RadioChannelModel, traffic_generator: TrafficGenerator):
        raise NotImplementedError("Subclasses must implement this method.")
    
class QueueAwareProportionalFairPhysicalResourceBlockScheduler(PhysicalResourceBlockScheduler):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        super().__init__(num_sectors, num_devices)

    def schedule(self, sector_manager: SectorManager, radio_channel_model: RadioChannelModel, traffic_generator: TrafficGenerator):
        pass