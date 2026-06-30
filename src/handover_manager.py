import numpy as np
from .radio_channel_model import RadioChannelModel
from .sector_manager import SectorManager
from .device_manager import DeviceManager
from .sleep_mode_manager import SleepModeManager


class HandoverManager:
    def __init__(self, num_sectors: int, num_devices: int, sleep_mode_manager: SleepModeManager | None = None) -> None:
        self.device_attachment_adjacency_matrix: np.ndarray = np.zeros(
            (num_sectors, num_devices), dtype=bool,
        )
        self.sleep_mode_manager = sleep_mode_manager

    def handover_event(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        radio_channel_model: RadioChannelModel,
    ) -> None:
        if radio_channel_model.downlink_received_signal_power_dbm_matrix is None:
            raise ValueError("Radio channel model must have received power matrix updated before handover event")

        if radio_channel_model.downlink_received_signal_power_dbm_matrix.shape != self.device_attachment_adjacency_matrix.shape:
            raise ValueError("Radio channel model matrix shape does not match attachment matrix shape")

        rx_power = radio_channel_model.downlink_received_signal_power_dbm_matrix.astype(np.float32)
        if self.sleep_mode_manager is not None:
            sleep_mask = self.sleep_mode_manager.sector_sleep_mode_vector == 3
            rx_power[sleep_mask, :] = -np.inf

        best_sector_per_device = np.argmax(rx_power, axis=0)
        self.device_attachment_adjacency_matrix.fill(False)
        self.device_attachment_adjacency_matrix[best_sector_per_device, np.arange(self.device_attachment_adjacency_matrix.shape[1], dtype=np.int32)] = True

    def handover_device(self, device_id: int, sector_id: int) -> None:
        device_id = int(device_id)
        sector_id = int(sector_id)
        self.device_attachment_adjacency_matrix[:, device_id] = False
        self.device_attachment_adjacency_matrix[sector_id, device_id] = True

    def is_device_attached(self, device_id: int, sector_id: int) -> bool:
        return bool(self.device_attachment_adjacency_matrix[sector_id, device_id])

    def get_attached_sector(self, device_id: int) -> int | None:
        device_id = int(device_id)
        attached_sectors = np.flatnonzero(self.device_attachment_adjacency_matrix[:, device_id])
        return int(attached_sectors[0]) if attached_sectors.size > 0 else None

    def get_attached_devices(self, sector_id: int) -> np.ndarray:
        sector_id = int(sector_id)
        return np.flatnonzero(self.device_attachment_adjacency_matrix[sector_id, :])
