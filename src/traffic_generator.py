import numpy as np
from abc import ABC, abstractmethod

class TrafficGenerator(ABC):
    def __init__(self, num_devices: int, window_length: int) -> None:
        self.num_devices = num_devices
        self.window_length = window_length
        self.device_downlink_bits_matrix: np.ndarray = np.zeros((num_devices, window_length), dtype=np.float32)

    @abstractmethod
    def generate_device_downlink_bits_matrix(self):
        raise NotImplementedError("Subclasses must implement this method.")
        
class ParetoDistributionTrafficGenerator(TrafficGenerator):
    def __init__(self, num_devices: int, window_length: int, shape: float = 1.5, scale: float = 1.0) -> None:
        super().__init__(num_devices, window_length)
        self.shape = shape
        self.scale = scale

    def generate_device_downlink_bits_matrix(self):
        self.device_downlink_bits_matrix = (np.random.pareto(self.shape, (self.num_devices, self.window_length)) + 1) * self.scale