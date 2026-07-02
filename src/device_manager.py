import numpy as np

class DeviceManager:
    def __init__(self, num_devices: int) -> None:
        self.device_position_matrix: np.ndarray = np.zeros((num_devices, 3), dtype=np.float32)
        self.device_velocity_matrix: np.ndarray = np.zeros((num_devices, 3), dtype=np.float32)
        velocity_row_values = np.array([0.001388,0.001388,0]) # 5 km/hr
        self.device_velocity_matrix[:] = velocity_row_values
        self.device_physical_resource_block_allocation_vector: np.ndarray = np.zeros((num_devices, ), dtype=np.int16)
        self.device_achievable_throughput_vector: np.ndarray = np.zeros((num_devices, ), dtype=np.int16)
        self.device_historic_throughput_vector: np.ndarray = np.zeros((num_devices, ), dtype=np.int16)

    def get_all_positions(self) -> np.ndarray:
        return self.device_position_matrix
    
    def get_all_velocities(self) -> np.ndarray:
        return self.device_velocity_matrix