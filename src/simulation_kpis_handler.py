import numpy as np
from time import time

class SimulationKPIHandler:
    def __init__(self, num_devices: int, num_sectors: int):
        self.num_devices = num_devices
        self.num_sectors = num_sectors
        self.total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)
        
        # Clock tracking attributes
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_time = 0.0

    def start_clock(self):
        """Starts the simulation timer."""
        self.start_time = time()

    def end_clock(self):
        """Stops the simulation timer."""
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        """Returns the total elapsed time in seconds."""
        return self.elapsed_time

    def update_kpis(self, total_transmitted_bits_per_device_vector):
        self.total_transmitted_bits_per_device += total_transmitted_bits_per_device_vector

    def calculate_throughput_mbps(self, total_simulated_ms):
        seconds = total_simulated_ms / 1000.0 + 1e-6
        return (self.total_transmitted_bits_per_device / 1e6) / seconds

    def calculate_average_throughput_mbps(self, total_ms):
        return np.mean(self.calculate_throughput_mbps(total_ms))

    def calculate_throughput_percentile(self, percentile, total_ms):
        return np.percentile(self.calculate_throughput_mbps(total_ms), percentile)

    def print_kpis(self, total_ms):
        avg = self.calculate_average_throughput_mbps(total_ms)
        p10 = self.calculate_throughput_percentile(10, total_ms)
        print(f"--- Simulation Performance ---")
        print(f"Wall-clock time: {self.get_elapsed_time():.2f} seconds")
        print(f"Average throughput (Mbps): {avg:.4f}")
        print(f"10th percentile throughput (Mbps): {p10:.4f}")