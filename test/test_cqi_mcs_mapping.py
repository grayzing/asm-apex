import numpy as np

from src.base_station_manager import BaseStationManager
from src.device_manager import DeviceManager
from src.radio_channel_model import RadioChannelModel
from src.sector_manager import SectorManager


def test_cqi_and_mcs_mapping_up_to_qam256():
    sector_mgr = SectorManager(num_sectors=1, num_base_stations=1)
    sector_mgr.center_freq_ghz_matrix = np.array([3.5], dtype=np.float32)
    sector_mgr.tx_power_dbm_matrix = np.array([43.0], dtype=np.float32)
    sector_mgr.bandwidth_mhz_matrix = np.array([20.0], dtype=np.float32)
    sector_mgr.sector_parent_base_station_vector = np.array([0], dtype=np.int32)
    sector_mgr.sector_azimuth_angle_deg_matrix = np.array([0.0], dtype=np.float32)
    sector_mgr.horizontal_beamwidth_deg_matrix = np.array([65.0], dtype=np.float32)
    sector_mgr.vertical_beamwidth_deg_matrix = np.array([8.0], dtype=np.float32)
    sector_mgr.max_array_gain_matrix = np.array([30.0], dtype=np.float32)
    sector_mgr.downtilt_deg_matrix = np.array([0.0], dtype=np.float32)
    sector_mgr.front_to_back_ratio_matrix = np.array([30.0], dtype=np.float32)
    sector_mgr.sector_numerology_matrix = np.array([1], dtype=np.int8)
    sector_mgr.sector_physical_resource_block_utilization = np.array([0.5], dtype=np.float32)

    device_mgr = DeviceManager(num_devices=1)
    device_mgr.device_position_matrix = np.array([[100.0, 0.0, 1.5]], dtype=np.float32)

    bs_mgr = BaseStationManager(num_base_stations=1, num_sectors=1)
    bs_mgr.base_station_position_matrix = np.array([[0.0, 0.0, 25.0]], dtype=np.float32)

    model = RadioChannelModel(num_sectors=1, num_devices=1)
    model.update_downlink_received_signal_power_dbm_matrix(sector_mgr, device_mgr, bs_mgr, shadowing_seed=1)
    model.update_downlink_total_interference_dbm_matrix(sector_mgr, device_mgr, bs_mgr, interference_seed=2)
    model.update_signal_interference_noise_ratio_dbm_matrix(sector_mgr, device_mgr, bs_mgr, interference_seed=2)
    model.update_downlink_channel_quality_indicator_matrix()
    model.update_downlink_modulation_coding_scheme_matrix()

    assert model.downlink_channel_quality_indicator_matrix.shape == (1, 1)
    assert model.downlink_modulation_coding_scheme_matrix.shape == (1, 1)
    assert model.downlink_modulation_coding_scheme_matrix[0, 0] <= 28
    assert model.downlink_modulation_coding_scheme_matrix[0, 0] >= 0
