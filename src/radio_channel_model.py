import numpy as np
from .geometry_helper import GeometryHelper
from .device_manager import DeviceManager
from .sector_manager import SectorManager
from .base_station_manager import BaseStationManager
from abc import ABC, abstractmethod

class RadioChannelModel(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.thermal_noise_floor_dbm_per_hertz: float = -174.0
        self.receiver_noise_figure_db: float = 9.0

        self.downlink_received_signal_power_dbm_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.fast_fading_db_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.downlink_total_interference_dbm_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.downlink_signal_interference_noise_ratio_dbm_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.downlink_channel_quality_indicator_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.downlink_modulation_coding_scheme_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)

        self.directional_gain_db_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.path_loss_db_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.shadowing_db_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.los_boolean_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=bool)
        self.breakpoint_distance_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)

        self.geometry_helper: GeometryHelper = GeometryHelper(num_sectors, num_devices)

        self.signal_interference_noise_ratio_to_channel_quality_indicator_mapping: np.ndarray = np.zeros((16, 1), dtype=np.float16)
        self.channel_quality_indicator_to_modulation_coding_scheme_mapping: np.ndarray = np.zeros((16, 1), dtype=np.float16)

    def calculate_directional_gain_db(self, sector_id: int, device_id: int) -> float:
        return float(self.directional_gain_db_matrix[sector_id, device_id])

    def _calculate_prb_properties(self, sector_manager: SectorManager) -> tuple[np.ndarray, np.ndarray]:
        numerology = sector_manager.sector_numerology_matrix.astype(np.int32)
        scs_hz = 15000.0 * (2**numerology)
        prb_bandwidth_hz = 12.0 * scs_hz
        total_bandwidth_hz = sector_manager.bandwidth_mhz_matrix.astype(np.float32) * 1e6
        prb_count = np.maximum(np.floor(total_bandwidth_hz / prb_bandwidth_hz), 1).astype(np.int32)
        return prb_bandwidth_hz.astype(np.float32), prb_count

    def _calculate_thermal_noise_dbm(self, sector_manager: SectorManager) -> np.ndarray:
        prb_bandwidth_hz, _ = self._calculate_prb_properties(sector_manager)
        return (self.thermal_noise_floor_dbm_per_hertz + 10.0 * np.log10(prb_bandwidth_hz)).astype(np.float32)

    def _sinr_to_cqi(self, sinr_db: np.ndarray) -> np.ndarray:
        thresholds = np.array(
            [-5.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            dtype=np.float32,
        )
        bin_index = np.digitize(sinr_db, thresholds, right=False)
        bin_index = np.minimum(bin_index, thresholds.size - 1)
        return (bin_index + 1).astype(np.int32)

    def _cqi_to_mcs(self, cqi: np.ndarray) -> np.ndarray:
        mapping = np.array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            dtype=np.int32,
        )
        cqi_clipped = np.clip(cqi, 1, mapping.size)
        return mapping[cqi_clipped - 1].astype(np.int32)

    def _calculate_um_a_path_loss_matrices(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        shadowing_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        device_positions = device_manager.get_all_positions().astype(np.float32)
        bs_positions = base_station_manager.get_all_positions().astype(np.float32)

        parent_bs_vector = self.geometry_helper._resolve_sector_parent_base_station_vector(
            sector_manager,
            base_station_manager,
        )

        sector_bs_positions = bs_positions[parent_bs_vector, :]

        dx = device_positions[None, :, 0] - sector_bs_positions[:, None, 0]
        dy = device_positions[None, :, 1] - sector_bs_positions[:, None, 1]
        dz = device_positions[None, :, 2] - sector_bs_positions[:, None, 2]

        d2d = np.sqrt(dx**2 + dy**2)
        d3d = np.sqrt(d2d**2 + dz**2)
        d3d = np.maximum(d3d, 1.0)

        fc_ghz = sector_manager.center_freq_ghz_matrix[parent_bs_vector, None].astype(np.float32)
        tx_power_dbm = sector_manager.tx_power_dbm_matrix[parent_bs_vector, None].astype(np.float32)

        h_bs = sector_bs_positions[:, 2][:, None].astype(np.float32)
        h_ut = device_positions[None, :, 2].astype(np.float32)
        lambda_m = 3e8 / (fc_ghz * 1e9)
        d_bp = np.maximum(4.0 * h_bs * h_ut / lambda_m, 10.0).astype(np.float32)

        los_probability = np.minimum(18.0 / np.maximum(d2d, 0.1), 1.0) * (
            1.0 - np.exp(-d2d / 63.0)
        ) + np.exp(-d2d / 63.0)

        rng = np.random.default_rng(shadowing_seed)
        los_random = rng.random(size=d2d.shape)
        los_boolean = los_random < los_probability

        pl_los = np.where(
            d3d <= d_bp,
            28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz),
            40.0 * np.log10(d3d) + 7.8 - 18.0 * np.log10(np.maximum(h_bs, 1.0)) + 20.0 * np.log10(fc_ghz),
        )

        pl_nlos = (
            13.54
            + 39.08 * np.log10(d3d)
            + 20.0 * np.log10(fc_ghz)
            - 0.6 * (h_ut - 1.5)
        )
        pl_nlos = np.maximum(pl_nlos, pl_los)

        path_loss = np.where(los_boolean, pl_los, pl_nlos).astype(np.float32)

        shadow_sigma = np.where(los_boolean, 4.0, 6.0).astype(np.float32)
        shadowing = rng.normal(loc=0.0, scale=shadow_sigma).astype(np.float32)

        return path_loss, shadowing, los_boolean, d_bp

    def update_downlink_received_signal_power_dbm_matrix(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        shadowing_seed: int = 0,
    ) -> None:
        self.update_directional_gain_db_matrix(sector_manager, device_manager, base_station_manager)

        path_loss, shadowing, los_boolean, breakpoint_distance = self._calculate_um_a_path_loss_matrices(
            sector_manager,
            device_manager,
            base_station_manager,
            shadowing_seed,
        )

        self.path_loss_db_matrix = path_loss
        self.shadowing_db_matrix = shadowing
        self.los_boolean_matrix = los_boolean
        self.breakpoint_distance_matrix = breakpoint_distance.astype(np.float32)

        _, prb_count = self._calculate_prb_properties(sector_manager)
        tx_power_dbm_per_prb = (
            sector_manager.tx_power_dbm_matrix.astype(np.float32)
            - 10.0 * np.log10(prb_count.astype(np.float32))
        )[:, None]

        rng_fading = np.random.default_rng(shadowing_seed + 1)
        fast_fading = rng_fading.rayleigh(scale=1.0, size=self.path_loss_db_matrix.shape).astype(np.float32)
        self.fast_fading_db_matrix = 20.0 * np.log10(np.maximum(fast_fading, 1e-6)).astype(np.float32)

        self.downlink_received_signal_power_dbm_matrix = (
            tx_power_dbm_per_prb
            + self.directional_gain_db_matrix
            - self.path_loss_db_matrix
            + self.shadowing_db_matrix
            + self.fast_fading_db_matrix
        ).astype(np.float32)

    def update_downlink_total_interference_dbm_matrix(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        interference_seed: int = 0,
    ) -> None:
        if self.downlink_received_signal_power_dbm_matrix is None:
            self.update_downlink_received_signal_power_dbm_matrix(
                sector_manager,
                device_manager,
                base_station_manager,
                shadowing_seed=interference_seed,
            )

        rng = np.random.default_rng(interference_seed)
        utilization = sector_manager.sector_physical_resource_block_utilization.astype(np.float32)[:, None]
        active = rng.random(size=(self.num_sectors, self.num_devices)).astype(np.float32) < utilization

        signal_linear = 10.0 ** ((self.downlink_received_signal_power_dbm_matrix.astype(np.float32) + 30.0) / 10.0)
        total_linear_per_device = np.sum(signal_linear * active, axis=0)

        interference_linear = total_linear_per_device[None, :] - signal_linear * active
        interference_linear = np.maximum(interference_linear, 1e-15)

        self.downlink_total_interference_dbm_matrix = (
            10.0 * np.log10(interference_linear) - 30.0
        ).astype(np.float32)

    def update_signal_interference_noise_ratio_dbm_matrix(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
        interference_seed: int = 0,
    ) -> None:
        self.update_downlink_total_interference_dbm_matrix(
            sector_manager,
            device_manager,
            base_station_manager,
            interference_seed=interference_seed,
        )

        noise_dbm = self._calculate_thermal_noise_dbm(sector_manager)[:, None] + self.receiver_noise_figure_db
        noise_linear = 10.0 ** ((noise_dbm + 30.0) / 10.0)

        signal_linear = 10.0 ** ((self.downlink_received_signal_power_dbm_matrix.astype(np.float32) + 30.0) / 10.0)
        interference_linear = 10.0 ** ((self.downlink_total_interference_dbm_matrix.astype(np.float32) + 30.0) / 10.0)

        sinr_linear = signal_linear / (interference_linear + noise_linear)
        self.downlink_signal_interference_noise_ratio_dbm_matrix = (
            10.0 * np.log10(np.maximum(sinr_linear, 1e-12))
        ).astype(np.float32)

    def update_downlink_channel_quality_indicator_matrix(self) -> None:
        self.downlink_channel_quality_indicator_matrix = self._sinr_to_cqi(
            self.downlink_signal_interference_noise_ratio_dbm_matrix.astype(np.float32)
        ).astype(np.float32)

    def update_downlink_modulation_coding_scheme_matrix(self) -> None:
        self.downlink_modulation_coding_scheme_matrix = self._cqi_to_mcs(
            self.downlink_channel_quality_indicator_matrix.astype(np.int32)
        ).astype(np.float32)

    def calculate_bits_per_prb_matrix(self) -> np.ndarray:
        mcs = self.downlink_modulation_coding_scheme_matrix.astype(np.float32)
        bits_matrix = 100.0 + mcs * 30.0
        return bits_matrix.astype(np.float32)

    def update_directional_gain_db_matrix(
        self,
        sector_manager: SectorManager,
        device_manager: DeviceManager,
        base_station_manager: BaseStationManager,
    ) -> None:
        self.geometry_helper.update_relative_azimuth_matrix(device_manager, sector_manager, base_station_manager)
        self.geometry_helper.update_relative_zenith_matrix(device_manager, sector_manager, base_station_manager)

        relative_azimuth = self.geometry_helper.relative_azimuth_angle_deg_matrix.astype(np.float32)
        relative_zenith = self.geometry_helper.relative_zenith_angle_deg_matrix.astype(np.float32)

        horizontal_3db = sector_manager.horizontal_beamwidth_deg_matrix[:, np.newaxis].astype(np.float32)
        vertical_3db = sector_manager.vertical_beamwidth_deg_matrix[:, np.newaxis].astype(np.float32)
        max_attenuation = sector_manager.front_to_back_ratio_matrix[:, np.newaxis].astype(np.float32)
        slant_loss = np.full((self.num_sectors, 1), 30.0, dtype=np.float32)
        max_gain = sector_manager.max_array_gain_matrix[:, np.newaxis].astype(np.float32)

        azim_ratio = np.abs(relative_azimuth) / np.maximum(horizontal_3db, 1e-6)
        elev_ratio = np.abs(relative_zenith) / np.maximum(vertical_3db, 1e-6)

        horizontal_attenuation = np.minimum(12.0 * azim_ratio**2, max_attenuation)
        vertical_attenuation = np.minimum(12.0 * elev_ratio**2, slant_loss)
        total_attenuation = np.minimum(horizontal_attenuation + vertical_attenuation, max_attenuation)

        self.directional_gain_db_matrix = (max_gain - total_attenuation).astype(np.float32)

