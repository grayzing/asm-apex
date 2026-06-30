import logging
from typing import Optional

import numpy as np

from .base_station_manager import BaseStationManager
from .device_manager import DeviceManager
from .geometry_helper import GeometryHelper
from .handover_manager import HandoverManager
from .network_topology_helper import NetworkTopologyHelper
from .mobility_helper import MobilityHelper
from .radio_channel_model import RadioChannelModel
from .scheduler import Scheduler
from .sector_manager import SectorManager
from .sleep_mode_manager import SleepModeManager
from .traffic_model import TrafficModel


logger = logging.getLogger(__name__)


class Simulation:
    def __init__(
        self,
        num_gnbs: int,
        num_devices: int,
        num_sectors_per_gnb: int = 3,
        topology_helper: Optional[NetworkTopologyHelper] = None,
        mobility_helper: Optional[MobilityHelper] = None,
        traffic_model: Optional[TrafficModel] = None,
        radio_channel_model: Optional[RadioChannelModel] = None,
        scheduler: Optional[Scheduler] = None,
        handover_manager: Optional[HandoverManager] = None,
        sleep_mode_manager: Optional[SleepModeManager] = None,
        seed: int = 0,
    ) -> None:
        self.num_gnbs = int(num_gnbs)
        self.num_devices = int(num_devices)
        self.num_sectors_per_gnb = int(num_sectors_per_gnb)
        self.num_sectors = self.num_gnbs * self.num_sectors_per_gnb
        self.seed = int(seed)

        self.base_station_manager = BaseStationManager(num_base_stations=self.num_gnbs, num_sectors=self.num_sectors)
        self.device_manager = DeviceManager(num_devices=self.num_devices)
        self.sector_manager = SectorManager(num_sectors=self.num_sectors, num_base_stations=self.num_gnbs)
        self.topology_helper = topology_helper
        self.mobility_helper = mobility_helper
        self.traffic_model = traffic_model
        self.radio_channel_model = radio_channel_model
        self.handover_manager = handover_manager
        self.sleep_mode_manager = sleep_mode_manager
        self.scheduler = scheduler

        self.logs: dict[str, list[float]] = {
            "sinr": [],
            "total_throughput": [],
            "average_throughput": [],
            "p10_throughput": [],
            "total_throughput_mbps": [],
            "average_throughput_mbps": [],
            "p10_throughput_mbps": [],
        }

    def setup_network(self) -> None:
        if self.topology_helper is None:
            raise ValueError("topology_helper is required")
        if self.radio_channel_model is None:
            raise ValueError("radio_channel_model is required")
        if self.traffic_model is None:
            raise ValueError("traffic_model is required")
        if self.scheduler is None:
            raise ValueError("scheduler is required")
        if self.handover_manager is None:
            raise ValueError("handover_manager is required")

        self.topology_helper.setup_network(
            self.device_manager,
            self.base_station_manager,
            self.sector_manager,
        )

        rng = np.random.default_rng(self.seed)
        if self.device_manager.device_position_matrix.shape[0] > 0:
            site_ids = rng.integers(0, self.num_gnbs, size=self.num_devices, dtype=np.int32)
            radii = rng.uniform(150.0, 500.0, size=self.num_devices).astype(np.float32)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=self.num_devices).astype(np.float32)
            x = self.base_station_manager.base_station_position_matrix[site_ids, 0] + radii * np.cos(angles)
            y = self.base_station_manager.base_station_position_matrix[site_ids, 1] + radii * np.sin(angles)
            self.device_manager.device_position_matrix[:, 0] = x
            self.device_manager.device_position_matrix[:, 1] = y
            self.device_manager.device_position_matrix[:, 2] = 1.5
            speeds = rng.uniform(0.5, 2.0, size=self.num_devices).astype(np.float32)
            headings = rng.uniform(0.0, 2.0 * np.pi, size=self.num_devices).astype(np.float32)
            self.device_manager.device_velocity_matrix[:, 0] = speeds * np.cos(headings)
            self.device_manager.device_velocity_matrix[:, 1] = speeds * np.sin(headings)
            self.device_manager.device_velocity_matrix[:, 2] = 0.0

        self.handover_manager.device_attachment_adjacency_matrix = np.zeros(
            (self.num_sectors, self.num_devices),
            dtype=bool,
        )
        self.radio_channel_model.update_downlink_received_signal_power_dbm_matrix(
            self.sector_manager,
            self.device_manager,
            self.base_station_manager,
            shadowing_seed=self.seed,
        )
        self.handover_manager.handover_event(
            self.sector_manager,
            self.device_manager,
            self.radio_channel_model,
        )

        logger.info("Network setup complete: %d gNBs, %d sectors, %d devices", self.num_gnbs, self.num_sectors, self.num_devices)

    def step(self, frame_index: int, shadowing_seed: int = 0, interference_seed: int = 0) -> None:
        if frame_index < 0 or frame_index >= self.traffic_model.time_frame_ms:
            raise IndexError("frame_index out of range")

        if self.mobility_helper is not None:
            self.mobility_helper.step_devices(self.device_manager)
            logger.debug("Mobility step applied for frame %d", frame_index)

        self.traffic_model.generate_bits_in_queue(frame_index=frame_index)
        logger.debug("Traffic generated for frame %d", frame_index)

        self.radio_channel_model.update_downlink_received_signal_power_dbm_matrix(
            self.sector_manager,
            self.device_manager,
            self.base_station_manager,
            shadowing_seed=shadowing_seed,
        )
        logger.debug("Received signal power updated for frame %d", frame_index)

        self.handover_manager.handover_event(
            self.sector_manager,
            self.device_manager,
            self.radio_channel_model,
        )
        logger.debug("Handover event processed for frame %d", frame_index)

        self.radio_channel_model.update_signal_interference_noise_ratio_dbm_matrix(
            self.sector_manager,
            self.device_manager,
            self.base_station_manager,
            interference_seed=interference_seed,
        )
        self.radio_channel_model.update_downlink_channel_quality_indicator_matrix()
        self.radio_channel_model.update_downlink_modulation_coding_scheme_matrix()
        logger.debug("SINR/CQI/MCS updated for frame %d", frame_index)

        per_prb_bits_matrix = self.radio_channel_model.calculate_bits_per_prb_matrix()
        allocation = self.scheduler.allocate_prbs(
            frame_index=frame_index,
            sector_device_rates=self.radio_channel_model.downlink_channel_quality_indicator_matrix,
            per_prb_bits_matrix=per_prb_bits_matrix,
        )
        logger.debug("PRB allocation complete for frame %d", frame_index)

        sinr_values = self.radio_channel_model.downlink_signal_interference_noise_ratio_dbm_matrix.flatten().astype(np.float32)
        per_device_throughput = np.sum(allocation.astype(np.float32) * per_prb_bits_matrix, axis=0)
        total_throughput = float(np.sum(per_device_throughput))
        average_throughput = float(total_throughput / max(self.num_devices, 1))
        p10_throughput = float(np.percentile(per_device_throughput, 10.0))

        total_throughput_mbps = total_throughput / 1000.0
        average_throughput_mbps = average_throughput / 1000.0
        p10_throughput_mbps = p10_throughput / 1000.0

        self.logs["sinr"].append(float(np.nanmean(sinr_values)))
        self.logs["total_throughput"].append(total_throughput)
        self.logs["average_throughput"].append(average_throughput)
        self.logs["p10_throughput"].append(p10_throughput)
        self.logs["total_throughput_mbps"].append(total_throughput_mbps)
        self.logs["average_throughput_mbps"].append(average_throughput_mbps)
        self.logs["p10_throughput_mbps"].append(p10_throughput_mbps)

        logger.info(
            "Frame %d: mean SINR=%.2f dB total throughput=%.1f bits avg=%.1f bits p10=%.1f bits (%.3f Mbps avg %.3f Mbps p10)",
            frame_index,
            float(np.nanmean(sinr_values)),
            total_throughput,
            average_throughput,
            p10_throughput,
            total_throughput_mbps,
            average_throughput_mbps,
        )

    def run(self, start_frame: int = 0, end_frame: int | None = None) -> None:
        if end_frame is None:
            end_frame = self.traffic_model.time_frame_ms

        for frame in range(start_frame, end_frame):
            self.step(frame_index=frame, shadowing_seed=frame, interference_seed=frame + 100)

    def get_metrics(self) -> dict[str, np.ndarray]:
        return {key: np.array(values, dtype=np.float32) for key, values in self.logs.items()}
