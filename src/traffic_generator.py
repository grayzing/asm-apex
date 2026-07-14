import numpy as np
from abc import ABC, abstractmethod

class TrafficGenerator(ABC):
    def __init__(self, num_devices: int, window_length: int, seed: int = 24) -> None:
        self.num_devices = num_devices
        self.window_length = window_length
        self.device_downlink_bits_matrix: np.ndarray = np.zeros((num_devices, window_length), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_device_downlink_bits_matrix(self):
        raise NotImplementedError("Subclasses must implement this method.")

class BurstyTrafficGenerator(TrafficGenerator):
    def __init__(self, num_devices, window_length, 
                 on_shape=2.5, off_shape=1.5, 
                 high_rate=3e6, low_rate=0, seed=24):
        super().__init__(num_devices, window_length, seed)
        self.on_shape = on_shape
        self.off_shape = off_shape
        self.high_rate = high_rate
        self.low_rate = low_rate

    def generate_device_downlink_bits_matrix(self):
        for i in range(self.num_devices):
            t = 0
            while t < self.window_length:
                on_duration = int(self.rng.pareto(self.on_shape) * 10 + 5)
                off_duration = int(self.rng.pareto(self.off_shape) * 50 + 10)
                
                end_on = min(t + on_duration, self.window_length)
                self.device_downlink_bits_matrix[i, t:end_on] = self.high_rate
                t = end_on
                
                if t < self.window_length:
                    end_off = min(t + off_duration, self.window_length)
                    self.device_downlink_bits_matrix[i, t:end_off] = self.low_rate
                    t = end_off
        
        return self.device_downlink_bits_matrix