import numpy as np

class SectorManager:
    def __init__(self, num_sectors: int):
        self.horizontal_beamwidth_deg_matrix: np.ndarray = np.full((num_sectors, ), 65.0, dtype=np.float16)
        self.vertical_beamwidth_deg_matrix: np.ndarray = np.full((num_sectors, ), 65.0, dtype=np.float16)
        self.downtilt_deg_matrix: np.ndarray = np.zeros((num_sectors, ), dtype=np.float16)
        self.front_to_back_ratio_matrix: np.ndarray = np.full((num_sectors, ), 30.0, dtype=np.float16)
        self.max_array_gain_matrix: np.ndarray = np.full((num_sectors, ), 30.0, dtype=np.float16)
        self.sector_azimuth_angle_deg_matrix: np.ndarray = np.resize([30, 150, 270], num_sectors)

        self.center_freq_ghz_matrix: np.ndarray = np.full((num_sectors, ), 6.0, dtype=np.float32)
        self.bandwidth_mhz_matrix: np.ndarray = np.full((num_sectors, ), 20.0, dtype=np.float32)
        self.tx_power_dbm_matrix: np.ndarray = np.full((num_sectors, ), 49.0, dtype=np.float32) # This is the Tx power in dBm over the entire bandwidth.
        self.sector_numerology_matrix: np.ndarray = np.full((num_sectors, ), 1, dtype=np.int8)
        self.sector_physical_resource_block_utilization: np.ndarray = np.full((num_sectors, ), 0.05, dtype=np.float32)

        self.sector_parent_base_station_vector: np.ndarray = np.zeros((num_sectors, ), dtype=np.int16)
