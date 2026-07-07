import numpy as np
from time import time

class SimulationKPIHandler:
    def __init__(self, num_devices: int):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

        self.num_devices = num_devices
        self.total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)
        self.total_prbs_allocated_per_device = np.zeros(self.num_devices, dtype=np.int64)

    def start_clock(self):
        self.start_time = time()

    def end_clock(self):
        self.end_time = time()

    def calculate_time_in_seconds(self):
        self.elapsed_time = self.end_time - self.start_time

    def update_kpis(self, total_transmitted_bits_per_device, total_prbs_allocated_per_device):
        self.total_transmitted_bits_per_device += total_transmitted_bits_per_device
        self.total_prbs_allocated_per_device += total_prbs_allocated_per_device

    def calculate_throughput_mbps(self, total_simulated_miliseconds):
        return self.total_transmitted_bits_per_device / (total_simulated_seconds / 1e6)

    def get_throughput_at_device(self, device_id, step):
        return self.calculate_average_throughput_mbps(self, step)

    def calculate_average_throughput_mbps(self):
        return np.mean(self.total_transmitted_bits_per_device)

    def calculate_throughput_percentile(self, percentile):
        assert 0 < percentile and percentile <= 100, "Bad range for percentile. Accepted range is (0, 100]"
        return np.percentile(self.calculate_average_throughput_mbps(), percentile)

