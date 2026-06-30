import numpy as np
from typing import Any

from .traffic_model import TrafficModel
from .sector_manager import SectorManager
from .handover_manager import HandoverManager
from .sleep_mode_manager import SleepModeManager


class Scheduler:
    def __init__(
        self,
        sector_manager: SectorManager,
        traffic_model: TrafficModel,
        handover_manager: HandoverManager,
        num_devices: int,
        num_sectors: int,
        num_prbs_per_sector: int = 100,
        banned_devices: np.ndarray | None = None,
        sleep_mode_manager: SleepModeManager | None = None,
    ) -> None:
        self.sector_manager = sector_manager
        self.traffic_model = traffic_model
        self.handover_manager = handover_manager
        self.num_devices = int(num_devices)
        self.num_sectors = int(num_sectors)
        self.num_prbs_per_sector = int(num_prbs_per_sector)
        self.banned_devices = np.asarray(banned_devices, dtype=np.int32) if banned_devices is not None else np.array([], dtype=np.int32)
        self.sleep_mode_manager = sleep_mode_manager

        self.device_prb_allocation_matrix = np.zeros(
            (self.num_sectors, self.num_devices),
            dtype=np.int16,
        )
        self.device_rate_estimate: np.ndarray = np.zeros((self.num_devices,), dtype=np.float32)

    def _compute_device_priorities(
        self,
        device_rates: np.ndarray,
        queue_sizes: np.ndarray,
    ) -> np.ndarray:
        device_rates = np.asarray(device_rates, dtype=np.float32)
        queue_sizes = np.asarray(queue_sizes, dtype=np.float32)
        fair_shares = np.maximum(device_rates, 1.0)
        priority = queue_sizes / fair_shares
        priority[queue_sizes <= 0] = 0.0
        return priority

    def allocate_prbs(
        self,
        frame_index: int,
        sector_device_rates: np.ndarray,
        per_prb_bits_matrix: np.ndarray | None = None,
        device_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.traffic_model.time_frame_ms:
            raise IndexError("frame_index out of range")

        if device_ids is None:
            device_ids = np.arange(self.num_devices, dtype=np.int32)
        else:
            device_ids = np.asarray(device_ids, dtype=np.int32)

        sector_device_rates = np.asarray(sector_device_rates, dtype=np.float32)
        if sector_device_rates.shape != (self.num_sectors, self.num_devices):
            raise ValueError("sector_device_rates must be shape (num_sectors, num_devices)")

        self.device_prb_allocation_matrix.fill(0)
        prbs_available = np.full((self.num_sectors,), self.num_prbs_per_sector, dtype=np.int32)

        for sector_idx in range(self.num_sectors):
            if self.sleep_mode_manager is not None:
                sleep_mode = int(self.sleep_mode_manager.sector_sleep_mode_vector[sector_idx])
                if sleep_mode == 3:
                    continue

            attached_devices = self.handover_manager.get_attached_devices(sector_idx)
            if attached_devices.size == 0:
                continue

            candidate_devices = np.intersect1d(attached_devices, device_ids, assume_unique=True)
            if candidate_devices.size == 0:
                continue

            sector_device_rates_for_attached = sector_device_rates[sector_idx, candidate_devices]
            queue_sizes_attached = self.traffic_model.num_bits_in_queue[candidate_devices].astype(np.float32)
            device_rates_attached = self.device_rate_estimate[candidate_devices]
            priorities_attached = self._compute_device_priorities(device_rates_attached, queue_sizes_attached)

            metric = priorities_attached * sector_device_rates_for_attached
            banned_mask = np.isin(candidate_devices, self.banned_devices)
            metric[banned_mask] = -np.inf

            valid_indices = np.flatnonzero(metric > 0.0)
            if valid_indices.size == 0:
                continue

            order = np.argsort(metric[valid_indices], kind="mergesort")[::-1]
            selected_devices = candidate_devices[valid_indices[order]]

            num_allocated = min(selected_devices.size, prbs_available[sector_idx])
            allocation = np.zeros((self.num_devices,), dtype=np.int16)
            allocation[selected_devices[:num_allocated]] = 1
            self.device_prb_allocation_matrix[sector_idx] = allocation

        if per_prb_bits_matrix is None:
            allocated_bits = self.device_prb_allocation_matrix.sum(axis=0).astype(np.int64) * self.sector_manager.device_bits_per_prb
        else:
            per_prb_bits_matrix = np.asarray(per_prb_bits_matrix, dtype=np.float32)
            if per_prb_bits_matrix.shape != (self.num_sectors, self.num_devices):
                raise ValueError("per_prb_bits_matrix must be shape (num_sectors, num_devices)")
            allocated_bits = np.sum(self.device_prb_allocation_matrix.astype(np.float32) * per_prb_bits_matrix, axis=0).astype(np.int64)

        self.traffic_model.consume_bits(np.arange(self.num_devices, dtype=np.int32), allocated_bits)
        self.device_rate_estimate = np.maximum(self.device_rate_estimate * 0.9 + allocated_bits * 0.1, 1.0)

        return self.device_prb_allocation_matrix
