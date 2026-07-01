import numpy as np
from abc import ABC, abstractmethod
from device_manager import DeviceManager

class MobilityHelper(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step_devices(self, device_manager: DeviceManager) -> None:
        pass

class RandomWalkMobilityHelper(MobilityHelper):
    def __init__(self, seed=24) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)

    def step_devices(self, device_manager: DeviceManager) -> None:
        positions = device_manager.get_all_positions().astype(np.float32)
        velocities = device_manager.get_all_velocities().astype(np.float32)

        perturbation = self.rng.uniform(-0.5, 0.5, size=velocities.shape).astype(np.float32)
        velocities[:, :2] = velocities[:, :2] + perturbation[:, :2]

        max_speed = 3.0
        speed = np.linalg.norm(velocities[:, :2], axis=1)
        speed = np.maximum(speed, 1e-6)
        scale = np.minimum(max_speed / speed, 1.0)
        velocities[:, :2] *= scale[:, None]

        new_positions = positions + velocities
        new_positions[:, 2] = np.maximum(new_positions[:, 2], 1.0)

        device_manager.device_velocity_matrix = velocities
        device_manager.device_position_matrix = new_positions
    