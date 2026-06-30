import numpy as np
from abc import ABC, abstractmethod


class TrafficModel(ABC):
    def __init__(
        self,
        time_frame_ms: int,
        num_devices: int,
        num_rf_devices: int | None = None,
        seed: int = 0,
    ) -> None:
        self.time_frame_ms = int(time_frame_ms)
        self.num_devices = int(num_devices)
        self.num_rf_devices = int(num_rf_devices) if num_rf_devices is not None else self.num_devices
        self.random_number_generator = np.random.default_rng(seed)

        self.queue_size_matrix: np.ndarray = np.zeros((self.num_devices, self.time_frame_ms), dtype=np.int64)
        self.num_bits_in_queue: np.ndarray = np.zeros((self.num_devices,), dtype=np.int64)

    @abstractmethod
    def generate_bits_in_queue(self, frame_index: int) -> np.ndarray:
        raise NotImplementedError

    def consume_bits(self, device_ids: np.ndarray, bits: np.ndarray) -> None:
        device_ids = np.asarray(device_ids, dtype=np.int32)
        bits = np.asarray(bits, dtype=np.int64)
        self.num_bits_in_queue[device_ids] = np.maximum(
            self.num_bits_in_queue[device_ids] - bits,
            0,
        )


class ParetoDistributionTrafficModel(TrafficModel):
    def __init__(
        self,
        time_frame_ms: int,
        num_devices: int,
        alpha: float = 1.5,
        scale_bits: int = 100_000,
        max_bits_per_frame: int = 5_000_000,
        num_rf_devices: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(time_frame_ms, num_devices, num_rf_devices=num_rf_devices, seed=seed)
        self.alpha = float(alpha)
        self.scale_bits = int(scale_bits)
        self.max_bits_per_frame = int(max_bits_per_frame)

    def generate_bits_in_queue(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.time_frame_ms:
            raise IndexError("frame_index out of range")

        samples = self.random_number_generator.pareto(self.alpha, size=self.num_devices) + 1.0
        bits_generated = np.minimum(
            np.floor(samples * float(self.scale_bits)).astype(np.int64),
            self.max_bits_per_frame,
        )

        self.num_bits_in_queue = np.maximum(self.num_bits_in_queue + bits_generated, 0)
        self.queue_size_matrix[:, frame_index] = self.num_bits_in_queue
        return self.num_bits_in_queue
