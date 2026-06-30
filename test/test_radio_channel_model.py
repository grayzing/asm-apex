import numpy as np

from src.radio_channel_model import RadioChannelModel


class DummyDeviceManager:
    def __init__(self, positions: np.ndarray) -> None:
        self._positions = positions.astype(np.float32)

    def get_all_positions(self) -> np.ndarray:
        return self._positions


class DummySectorManager:
    def __init__(self) -> None:
        self.horizontal_beamwidth_deg_matrix = np.array([65.0, 65.0], dtype=np.float32)
        self.vertical_beamwidth_deg_matrix = np.array([8.0, 8.0], dtype=np.float32)
        self.downtilt_deg_matrix = np.array([0.0, 0.0], dtype=np.float32)
        self.front_to_back_ratio_matrix = np.array([30.0, 30.0], dtype=np.float32)
        self.max_array_gain_matrix = np.array([30.0, 30.0], dtype=np.float32)
        self.sector_azimuth_angle_deg_matrix = np.array([0.0, 90.0], dtype=np.float32)
        self.sector_parent_base_station_vector = np.array([0, 0], dtype=np.int32)


class DummyBaseStationManager:
    def __init__(self, positions: np.ndarray) -> None:
        self._positions = positions.astype(np.float32)

    def get_all_positions(self) -> np.ndarray:
        return self._positions


def test_update_directional_gain_db_matrix_vectorized():
    sector_mgr = DummySectorManager()
    device_mgr = DummyDeviceManager(np.array([[10.0, 0.0, 30.0], [0.0, 10.0, 30.0]]))
    bs_mgr = DummyBaseStationManager(np.array([[0.0, 0.0, 30.0]]))

    model = RadioChannelModel(num_sectors=2, num_devices=2)
    model.update_directional_gain_db_matrix(sector_mgr, device_mgr, bs_mgr) # type: ignore

    expected_azimuth_attenuation = np.minimum(12.0 * (90.0 / 65.0) ** 2, 30.0)
    expected_gain = np.array([
        [30.0, 30.0 - expected_azimuth_attenuation],
        [30.0 - expected_azimuth_attenuation, 30.0],
    ], dtype=np.float32)

    np.testing.assert_allclose(model.directional_gain_db_matrix, expected_gain, atol=1e-4)
    assert np.isclose(model.calculate_directional_gain_db(0, 1), float(expected_gain[0, 1]), atol=1e-4)
