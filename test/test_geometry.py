import sys
import pytest
import numpy as np

# ==========================================
# 1. DEFINE MOCK IMPLEMENTATIONS
# ==========================================

class MockDeviceManager:
    def __init__(self, positions: list) -> None:
        self.positions = np.array(positions, dtype=np.float32)
    def get_all_positions(self) -> np.ndarray:
        return self.positions

class MockBaseStationManager:
    def __init__(self, positions: list) -> None:
        self.positions = np.array(positions, dtype=np.float32)
    def get_all_positions(self) -> np.ndarray:
        return self.positions

class MockSectorManager:
    def __init__(self, parent_vector: list, azimuths: list = None, downtilts: list = None) -> None: # type: ignore
        self.sector_parent_base_station_vector = np.array(parent_vector, dtype=np.int32)
        if azimuths is not None:
            self.sector_azimuth_angle_deg_matrix = np.array(azimuths, dtype=np.float32)
        if downtilts is not None:
            self.downtilt_deg_matrix = np.array(downtilts, dtype=np.float32)

# ==========================================================
# 2. INJECT MOCKS INTO SYS.MODULES BEFORE IMPORTING GEOMETRY
# ==========================================================
# This tricks geometry_helper into using our local mock stubs 
# rather than looking for unwritten or heavy source managers.
original_modules = {
    'src.device_manager': sys.modules.get('src.device_manager'),
    'src.sector_manager': sys.modules.get('src.sector_manager'),
    'src.base_station_manager': sys.modules.get('src.base_station_manager'),
}

sys.modules['src.device_manager'] = sys.modules[__name__]
sys.modules['src.sector_manager'] = sys.modules[__name__]
sys.modules['src.base_station_manager'] = sys.modules[__name__]

# Bind names so the module-level imports inside geometry_helper resolve perfectly
DeviceManager = MockDeviceManager
SectorManager = MockSectorManager
BaseStationManager = MockBaseStationManager

try:
    # NOW it is safe to import GeometryHelper
    from src.geometry_helper import GeometryHelper
finally:
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

# ==========================================
# 3. UNIT TEST CASES (Update class initializations to use Mocks)
# ==========================================

def test_geometry_helper_initialization():
    geo = GeometryHelper(num_sectors=6, num_devices=20)
    assert geo.num_sectors == 6
    assert geo.num_devices == 20
    assert geo.distance_matrix_meters_matrix.shape == (6, 20)

def test_update_distance_matrix():
    bs_mgr = MockBaseStationManager([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    sector_mgr = MockSectorManager(parent_vector=[0, 1], azimuths=[0.0, 0.0], downtilts=[0.0, 0.0])
    device_mgr = MockDeviceManager([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    
    geo = GeometryHelper(num_sectors=4, num_devices=2)
    geo.update_distance_matrix(device_mgr, sector_mgr, bs_mgr) # type: ignore
    
    expected_distances = np.array([
        [5.0, 12.0],
        [5.0, 12.0],
        [np.sqrt(65), np.sqrt(244)],
        [np.sqrt(65), np.sqrt(244)]
    ], dtype=np.float32)
    
    np.testing.assert_allclose(geo.distance_matrix_meters_matrix, expected_distances, rtol=1e-5)

def test_update_relative_azimuth_matrix():
    bs_mgr = MockBaseStationManager([[0.0, 0.0, 0.0]])
    sector_mgr = MockSectorManager(parent_vector=[0, 1, 2], azimuths=[30.0, 150.0, 270.0], downtilts=[0.0, 0.0, 0.0])
    device_mgr = MockDeviceManager([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    
    geo = GeometryHelper(num_sectors=3, num_devices=2)
    geo.update_relative_azimuth_matrix(device_mgr, sector_mgr, bs_mgr) # type: ignore
    
    expected_azimuths = np.array([
        [-30.0,  60.0],
        [-150.0, -60.0],
        [-90.0,  -180.0]
    ], dtype=np.float32)
    
    np.testing.assert_allclose(geo.relative_azimuth_angle_deg_matrix, expected_azimuths, atol=1e-4)

def test_update_relative_zenith_matrix():
    bs_mgr = MockBaseStationManager([[0.0, 0.0, 30.0]])
    sector_mgr = MockSectorManager(parent_vector=[0, 1], azimuths=[0.0, 0.0], downtilts=[0.0, 10.0])
    device_mgr = MockDeviceManager([[0.0, 0.0, 0.0], [10.0, 0.0, 30.0]])
    
    geo = GeometryHelper(num_sectors=2, num_devices=2)
    geo.update_relative_zenith_matrix(device_mgr, sector_mgr, bs_mgr) # type: ignore
    
    expected_zeniths = np.array([
        [90.0,   0.0],
        [80.0, -10.0]
    ], dtype=np.float32)
    
    np.testing.assert_allclose(geo.relative_zenith_angle_deg_matrix, expected_zeniths, atol=1e-4)
